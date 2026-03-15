//! HDCMemory — in-memory index with SQLite persistence.
//!
//! Stores HyperVector episodes with importance, access count, and surprise.
//! Supports brute-force top-K query (XOR + popcount), parallel query via rayon,
//! and SQLite persistence for crash recovery.

use crate::vector::HyperVector;
use rusqlite;
use std::time::{SystemTime, UNIX_EPOCH};

/// A single memory episode.
#[derive(Clone, Debug)]
pub struct Episode {
    pub id: u64,
    pub vector: HyperVector,
    pub metadata: String,
    pub timestamp: u64,        // unix millis
    pub importance: f32,       // 0.0 to 1.0
    pub access_count: u32,
    pub surprise_score: f32,
}

/// Configuration for memory store.
#[derive(Clone, Debug)]
pub struct MemoryConfig {
    pub max_episodes: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self { max_episodes: 100_000 }
    }
}

/// Forgetting policy for memory maintenance.
#[derive(Clone, Debug)]
pub struct ForgettingPolicy {
    /// Decay rate λ: higher = forget faster.
    pub decay_rate: f32,
    /// Prune episodes below this effective importance.
    pub min_importance: f32,
    /// Merge episodes more similar than this threshold.
    pub merge_similarity: f32,
    /// Hard cap on episode count.
    pub max_episodes: usize,
}

impl Default for ForgettingPolicy {
    fn default() -> Self {
        Self {
            decay_rate: 0.001,
            min_importance: 0.01,
            merge_similarity: 0.85,
            max_episodes: 100_000,
        }
    }
}

/// Report from a maintenance cycle.
#[derive(Clone, Debug)]
pub struct MaintenanceReport {
    pub pruned: usize,
    pub merged: usize,
    pub remaining: usize,
}

/// HDC memory store with in-memory index and SQLite backend.
pub struct HDCMemory {
    /// In-memory episode index.
    episodes: Vec<Episode>,
    /// SQLite connection for persistence.
    db: rusqlite::Connection,
    /// Auto-incrementing ID counter.
    next_id: u64,
    /// Configuration.
    #[allow(dead_code)]
    config: MemoryConfig,
}

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS episodes (
    id           INTEGER PRIMARY KEY,
    vector       BLOB NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}',
    timestamp    INTEGER NOT NULL,
    importance   REAL NOT NULL DEFAULT 0.5,
    access_count INTEGER NOT NULL DEFAULT 0,
    surprise     REAL NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_importance ON episodes(importance);
CREATE INDEX IF NOT EXISTS idx_timestamp ON episodes(timestamp);
";

/// Get current timestamp in milliseconds.
fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

impl HDCMemory {
    /// Open or create memory database at given path.
    pub fn open(db_path: &str, config: MemoryConfig) -> Result<Self, rusqlite::Error> {
        let db = rusqlite::Connection::open(db_path)?;
        db.execute_batch(SCHEMA_SQL)?;
        let mut mem = Self {
            episodes: Vec::new(),
            db,
            next_id: 0,
            config,
        };
        mem.load_from_disk()?;
        Ok(mem)
    }

    /// Create in-memory database (for testing, no file).
    pub fn in_memory(config: MemoryConfig) -> Result<Self, rusqlite::Error> {
        Self::open(":memory:", config)
    }

    /// Store a new episode. Returns episode ID.
    pub fn insert(
        &mut self,
        vector: HyperVector,
        metadata: &str,
        importance: f32,
        surprise: f32,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let episode = Episode {
            id,
            vector,
            metadata: metadata.to_string(),
            timestamp: current_timestamp_ms(),
            importance: importance.clamp(0.0, 1.0),
            access_count: 0,
            surprise_score: surprise.max(0.0),
        };
        self.episodes.push(episode);
        id
    }

    /// Find top-K most similar episodes to query.
    ///
    /// Brute-force: XOR + popcount over all episodes.
    /// Returns Vec<(episode_id, similarity)> sorted by similarity descending.
    pub fn query_topk(&self, query: &HyperVector, k: usize) -> Vec<(u64, f32)> {
        if self.episodes.is_empty() || k == 0 {
            return Vec::new();
        }
        let mut scored: Vec<(u64, f32)> = self.episodes.iter()
            .map(|ep| (ep.id, query.similarity(&ep.vector)))
            .collect();

        let effective_k = k.min(scored.len());
        if effective_k < scored.len() {
            scored.select_nth_unstable_by(effective_k - 1,
                |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        scored.truncate(effective_k);
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Parallel query using rayon (for large stores).
    pub fn query_topk_parallel(&self, query: &HyperVector, k: usize) -> Vec<(u64, f32)> {
        use rayon::prelude::*;
        if self.episodes.is_empty() || k == 0 {
            return Vec::new();
        }
        let mut scored: Vec<(u64, f32)> = self.episodes.par_iter()
            .map(|ep| (ep.id, query.similarity(&ep.vector)))
            .collect();

        let effective_k = k.min(scored.len());
        if effective_k < scored.len() {
            scored.select_nth_unstable_by(effective_k - 1,
                |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        scored.truncate(effective_k);
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Get episode by ID.
    pub fn get(&self, id: u64) -> Option<&Episode> {
        self.episodes.iter().find(|ep| ep.id == id)
    }

    /// Update access count (called when episode is retrieved).
    pub fn touch(&mut self, id: u64) {
        if let Some(ep) = self.episodes.iter_mut().find(|ep| ep.id == id) {
            ep.access_count += 1;
        }
    }

    /// Number of stored episodes.
    pub fn len(&self) -> usize { self.episodes.len() }

    /// Whether memory is empty.
    pub fn is_empty(&self) -> bool { self.episodes.is_empty() }

    // ── Persistence ───────────────────────────────────────────

    /// Persist all episodes to SQLite.
    pub fn sync_to_disk(&mut self) -> Result<(), rusqlite::Error> {
        self.db.execute("DELETE FROM episodes", [])?;
        for ep in &self.episodes {
            self.db.execute(
                "INSERT INTO episodes (id, vector, metadata, timestamp, importance, access_count, surprise)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![
                    ep.id as i64,
                    ep.vector.to_bytes(),
                    ep.metadata,
                    ep.timestamp as i64,
                    ep.importance as f64,
                    ep.access_count as i64,
                    ep.surprise_score as f64,
                ],
            )?;
        }
        Ok(())
    }

    /// Load episodes from SQLite into memory.
    pub fn load_from_disk(&mut self) -> Result<(), rusqlite::Error> {
        let mut stmt = self.db.prepare(
            "SELECT id, vector, metadata, timestamp, importance, access_count, surprise FROM episodes"
        )?;
        self.episodes = stmt.query_map([], |row| {
            let id: i64 = row.get(0)?;
            let vector_bytes: Vec<u8> = row.get(1)?;
            let vector = HyperVector::from_bytes(&vector_bytes)
                .ok_or(rusqlite::Error::InvalidQuery)?;
            let ts: i64 = row.get(3)?;
            let imp: f64 = row.get(4)?;
            let ac: i64 = row.get(5)?;
            let sur: f64 = row.get(6)?;
            Ok(Episode {
                id: id as u64,
                vector,
                metadata: row.get(2)?,
                timestamp: ts as u64,
                importance: imp as f32,
                access_count: ac as u32,
                surprise_score: sur as f32,
            })
        })?.collect::<Result<Vec<_>, _>>()?;

        self.next_id = self.episodes.iter().map(|ep| ep.id).max().unwrap_or(0) + 1;
        Ok(())
    }

    // ── Forgetting & Maintenance ──────────────────────────────

    /// Compute effective importance for an episode.
    ///
    /// eff = importance × exp(-λ × age_seconds) × log(1 + access_count) × (1 + surprise)
    pub fn effective_importance(&self, episode: &Episode, policy: &ForgettingPolicy) -> f32 {
        let now = current_timestamp_ms();
        let age_seconds = (now.saturating_sub(episode.timestamp)) as f32 / 1000.0;
        let decay = (-policy.decay_rate * age_seconds).exp();
        let access_boost = (2.0 + episode.access_count as f32).ln();
        let surprise_boost = 1.0 + episode.surprise_score;
        episode.importance * decay * access_boost * surprise_boost
    }

    /// Run full maintenance cycle.
    ///
    /// 1. Prune episodes below min_importance
    /// 2. Merge similar episodes (similarity > merge_similarity)
    /// 3. Hard cap: drop lowest effective importance
    pub fn maintain(&mut self, policy: &ForgettingPolicy) -> MaintenanceReport {
        let mut pruned = 0usize;

        // Step 1: Prune low-importance
        let before = self.episodes.len();
        // Compute effective importance for each, keep those above threshold
        let min_imp = policy.min_importance;
        let decay_rate = policy.decay_rate;
        let now = current_timestamp_ms();
        self.episodes.retain(|ep| {
            let age_seconds = (now.saturating_sub(ep.timestamp)) as f32 / 1000.0;
            let decay = (-decay_rate * age_seconds).exp();
            let access_boost = (2.0 + ep.access_count as f32).ln();
            let surprise_boost = 1.0 + ep.surprise_score;
            let eff = ep.importance * decay * access_boost * surprise_boost;
            eff >= min_imp
        });
        pruned += before - self.episodes.len();

        // Step 2: Merge similar episodes (greedy O(n²))
        let merged = self.merge_similar(policy.merge_similarity);

        // Step 3: Hard cap
        if self.episodes.len() > policy.max_episodes {
            // Sort by effective importance descending, truncate
            self.episodes.sort_by(|a, b| {
                let ea = {
                    let age = (now.saturating_sub(a.timestamp)) as f32 / 1000.0;
                    a.importance * (-decay_rate * age).exp()
                        * (2.0 + a.access_count as f32).ln()
                        * (1.0 + a.surprise_score)
                };
                let eb = {
                    let age = (now.saturating_sub(b.timestamp)) as f32 / 1000.0;
                    b.importance * (-decay_rate * age).exp()
                        * (2.0 + b.access_count as f32).ln()
                        * (1.0 + b.surprise_score)
                };
                eb.partial_cmp(&ea).unwrap_or(std::cmp::Ordering::Equal)
            });
            let extra = self.episodes.len() - policy.max_episodes;
            self.episodes.truncate(policy.max_episodes);
            pruned += extra;
        }

        MaintenanceReport {
            pruned,
            merged,
            remaining: self.episodes.len(),
        }
    }

    /// Merge episodes with similarity > threshold.
    ///
    /// Merged vector = bundle(a, b), metadata combined,
    /// importance = max(a, b), access_count = sum.
    fn merge_similar(&mut self, threshold: f32) -> usize {
        if self.episodes.len() < 2 { return 0; }

        let mut merged_count = 0usize;
        let mut to_remove = Vec::new();

        // O(n²) pairwise comparison — acceptable for < 100K
        let n = self.episodes.len();
        for i in 0..n {
            if to_remove.contains(&i) { continue; }
            for j in (i + 1)..n {
                if to_remove.contains(&j) { continue; }
                let sim = self.episodes[i].vector.similarity(&self.episodes[j].vector);
                if sim > threshold {
                    // Merge j into i
                    let merged_vec = HyperVector::bundle(&[
                        &self.episodes[i].vector,
                        &self.episodes[j].vector,
                    ]);
                    self.episodes[i].vector = merged_vec;
                    self.episodes[i].importance = self.episodes[i].importance
                        .max(self.episodes[j].importance);
                    self.episodes[i].access_count += self.episodes[j].access_count;
                    self.episodes[i].surprise_score = self.episodes[i].surprise_score
                        .max(self.episodes[j].surprise_score);
                    // Combine metadata
                    if self.episodes[j].metadata != "{}" {
                        self.episodes[i].metadata = format!(
                            "{}+{}", self.episodes[i].metadata, self.episodes[j].metadata
                        );
                    }
                    to_remove.push(j);
                    merged_count += 1;
                }
            }
        }

        // Remove merged episodes (reverse order to preserve indices)
        to_remove.sort_unstable();
        for &idx in to_remove.iter().rev() {
            self.episodes.remove(idx);
        }

        merged_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn test_rng() -> StdRng { StdRng::seed_from_u64(12345) }

    #[test]
    fn test_in_memory_create() {
        let mem = HDCMemory::in_memory(MemoryConfig::default()).unwrap();
        assert_eq!(mem.len(), 0);
        assert!(mem.is_empty());
    }

    #[test]
    fn test_insert_and_query() {
        let mut mem = HDCMemory::in_memory(MemoryConfig::default()).unwrap();
        let mut rng = test_rng();

        let target = HyperVector::random(&mut rng);
        let other1 = HyperVector::random(&mut rng);
        let other2 = HyperVector::random(&mut rng);

        mem.insert(other1, "other1", 0.5, 0.0);
        let target_id = mem.insert(target.clone(), "target", 0.8, 0.0);
        mem.insert(other2, "other2", 0.5, 0.0);

        let results = mem.query_topk(&target, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, target_id, "Top result should be the target");
        assert!((results[0].1 - 1.0).abs() < f32::EPSILON, "Exact match should have sim=1.0");
    }

    #[test]
    fn test_topk_ordering() {
        let mut mem = HDCMemory::in_memory(MemoryConfig::default()).unwrap();
        let mut rng = test_rng();

        // Insert 50 random episodes
        for i in 0..50 {
            let v = HyperVector::random(&mut rng);
            mem.insert(v, &format!("ep{i}"), 0.5, 0.0);
        }

        // Insert one near-identical to query
        let query = HyperVector::random(&mut rng);
        let near_id = mem.insert(query.clone(), "near_match", 0.5, 0.0);

        let results = mem.query_topk(&query, 5);
        assert_eq!(results[0].0, near_id, "Exact match should be rank 1");
        // All results should be ordered
        for w in results.windows(2) {
            assert!(w[0].1 >= w[1].1, "Results should be sorted descending");
        }
    }

    #[test]
    fn test_touch_updates_access() {
        let mut mem = HDCMemory::in_memory(MemoryConfig::default()).unwrap();
        let mut rng = test_rng();
        let id = mem.insert(HyperVector::random(&mut rng), "test", 0.5, 0.0);

        assert_eq!(mem.get(id).unwrap().access_count, 0);
        mem.touch(id);
        assert_eq!(mem.get(id).unwrap().access_count, 1);
        mem.touch(id);
        assert_eq!(mem.get(id).unwrap().access_count, 2);
    }

    #[test]
    fn test_empty_query() {
        let mem = HDCMemory::in_memory(MemoryConfig::default()).unwrap();
        let mut rng = test_rng();
        let results = mem.query_topk(&HyperVector::random(&mut rng), 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_k_larger_than_store() {
        let mut mem = HDCMemory::in_memory(MemoryConfig::default()).unwrap();
        let mut rng = test_rng();
        for _ in 0..5 {
            mem.insert(HyperVector::random(&mut rng), "{}", 0.5, 0.0);
        }
        let results = mem.query_topk(&HyperVector::random(&mut rng), 10);
        assert_eq!(results.len(), 5, "Should return all 5 when k=10 > len=5");
    }

    #[test]
    fn test_persistence_roundtrip() {
        let mut mem = HDCMemory::in_memory(MemoryConfig::default()).unwrap();
        let mut rng = test_rng();

        let mut id_vec_pairs: Vec<(u64, HyperVector)> = Vec::new();
        for i in 0..20 {
            let v = HyperVector::random(&mut rng);
            let id = mem.insert(v.clone(), &format!("ep{i}"), 0.5 + i as f32 * 0.01, 0.0);
            id_vec_pairs.push((id, v));
        }
        assert_eq!(mem.len(), 20);

        // Sync to disk
        mem.sync_to_disk().unwrap();

        // Clear in-memory and reload
        mem.episodes.clear();
        assert_eq!(mem.len(), 0);

        mem.load_from_disk().unwrap();
        assert_eq!(mem.len(), 20);

        // Verify vectors are byte-exact using stored IDs
        for (id, orig) in &id_vec_pairs {
            let ep = mem.get(*id).expect(&format!("Episode {id} should exist after reload"));
            assert_eq!(&ep.vector, orig, "Vector {id} should match after roundtrip");
        }
    }

    #[test]
    fn test_parallel_query_matches_serial() {
        let mut mem = HDCMemory::in_memory(MemoryConfig::default()).unwrap();
        let mut rng = test_rng();
        for _ in 0..100 {
            mem.insert(HyperVector::random(&mut rng), "{}", 0.5, 0.0);
        }
        let query = HyperVector::random(&mut rng);
        let serial = mem.query_topk(&query, 10);
        let parallel = mem.query_topk_parallel(&query, 10);
        assert_eq!(serial.len(), parallel.len());
        for (s, p) in serial.iter().zip(parallel.iter()) {
            assert_eq!(s.0, p.0, "Same episode IDs");
            assert!((s.1 - p.1).abs() < 1e-6, "Same similarities");
        }
    }

    #[test]
    fn test_prune_low_importance() {
        let mut mem = HDCMemory::in_memory(MemoryConfig::default()).unwrap();
        let mut rng = test_rng();

        // 7 high-importance, 3 low-importance (will decay below threshold)
        for _ in 0..7 {
            mem.insert(HyperVector::random(&mut rng), "{}", 0.8, 0.0);
        }
        for _ in 0..3 {
            // Very low importance + faked old timestamp
            let id = mem.insert(HyperVector::random(&mut rng), "{}", 0.001, 0.0);
            // Make it old by setting timestamp far in the past
            if let Some(ep) = mem.episodes.iter_mut().find(|e| e.id == id) {
                ep.timestamp = 0; // epoch = very old
            }
        }

        let policy = ForgettingPolicy {
            decay_rate: 0.001,
            min_importance: 0.005,
            merge_similarity: 0.99, // high → effectively no merging
            max_episodes: 100_000,
        };
        let report = mem.maintain(&policy);
        assert_eq!(report.remaining, 7, "Should prune the 3 low-importance episodes");
        assert!(report.pruned >= 3);
    }

    #[test]
    fn test_hard_cap() {
        let mut mem = HDCMemory::in_memory(MemoryConfig::default()).unwrap();
        let mut rng = test_rng();
        for _ in 0..100 {
            mem.insert(HyperVector::random(&mut rng), "{}", 0.5, 0.0);
        }
        let policy = ForgettingPolicy {
            decay_rate: 0.0, // no decay
            min_importance: 0.0, // no pruning
            merge_similarity: 0.999, // no merging
            max_episodes: 50,
        };
        let report = mem.maintain(&policy);
        assert_eq!(mem.len(), 50, "Hard cap should enforce max_episodes=50");
        assert_eq!(report.remaining, 50);
        assert!(report.pruned >= 50);
    }

    #[test]
    fn test_merge_similar() {
        let mut mem = HDCMemory::in_memory(MemoryConfig::default()).unwrap();
        let mut rng = test_rng();

        // Insert two identical vectors → should merge
        let v = HyperVector::random(&mut rng);
        mem.insert(v.clone(), "A", 0.5, 0.1);
        mem.insert(v.clone(), "B", 0.3, 0.2);
        // Insert one different
        mem.insert(HyperVector::random(&mut rng), "C", 0.5, 0.0);

        assert_eq!(mem.len(), 3);
        let policy = ForgettingPolicy {
            decay_rate: 0.0,
            min_importance: 0.0,
            merge_similarity: 0.9,
            max_episodes: 100_000,
        };
        let report = mem.maintain(&policy);
        assert_eq!(report.merged, 1, "Should merge the two identical episodes");
        assert_eq!(mem.len(), 2, "2 episodes should remain after merge");
    }

    #[test]
    fn test_maintenance_report() {
        let mut mem = HDCMemory::in_memory(MemoryConfig::default()).unwrap();
        let mut rng = test_rng();
        for _ in 0..10 {
            mem.insert(HyperVector::random(&mut rng), "{}", 0.5, 0.0);
        }
        let policy = ForgettingPolicy::default();
        let report = mem.maintain(&policy);
        assert_eq!(report.remaining, mem.len());
    }

    #[test]
    fn test_1k_scale() {
        let mut mem = HDCMemory::in_memory(MemoryConfig::default()).unwrap();
        let mut rng = test_rng();
        for _ in 0..1000 {
            mem.insert(HyperVector::random(&mut rng), "{}", 0.5, 0.0);
        }
        let query = HyperVector::random(&mut rng);
        let results = mem.query_topk(&query, 32);
        assert_eq!(results.len(), 32);
        for (_, sim) in &results {
            assert!(*sim >= 0.0 && *sim <= 1.0);
        }
    }

    #[test]
    fn test_10k_scale_timing() {
        let mut mem = HDCMemory::in_memory(MemoryConfig::default()).unwrap();
        let mut rng = test_rng();
        for _ in 0..10_000 {
            mem.insert(HyperVector::random(&mut rng), "{}", 0.5, 0.0);
        }
        let query = HyperVector::random(&mut rng);

        let t0 = std::time::Instant::now();
        let results = mem.query_topk(&query, 32);
        let elapsed = t0.elapsed();

        assert_eq!(results.len(), 32);
        eprintln!("10K query_topk(32): {:?}", elapsed);
        assert!(elapsed.as_millis() < 50, "10K query should be < 50ms, was {:?}", elapsed);
    }
}
