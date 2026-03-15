//! Causal graph — DAG of cause→effect relationships.
//!
//! Uses petgraph for graph operations. Supports topological reasoning,
//! intervention analysis ("what if X changes?"), and path queries.

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo;
use std::collections::HashMap;

/// A node in the causal graph.
#[derive(Clone, Debug)]
pub struct CausalNode {
    /// Human-readable label.
    pub label: String,
    /// Current state value.
    pub value: f32,
    /// Confidence in the causal relationship.
    pub confidence: f32,
}

/// Edge weight: strength and type of causal relationship.
#[derive(Clone, Debug)]
pub struct CausalEdge {
    /// Causal strength (positive = promotes, negative = inhibits).
    pub strength: f32,
    /// Lag in time steps.
    pub lag: u32,
}

/// Causal graph for structured world modeling.
#[derive(Clone)]
pub struct CausalGraph {
    /// Internal petgraph DAG.
    graph: DiGraph<CausalNode, CausalEdge>,
    /// Label → NodeIndex lookup.
    label_to_idx: HashMap<String, NodeIndex>,
}

impl CausalGraph {
    /// Create empty graph.
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            label_to_idx: HashMap::new(),
        }
    }

    /// Add a causal node. Returns node index.
    pub fn add_node(&mut self, label: &str, value: f32) -> NodeIndex {
        if let Some(&idx) = self.label_to_idx.get(label) {
            // Update existing
            if let Some(node) = self.graph.node_weight_mut(idx) {
                node.value = value;
            }
            return idx;
        }
        let idx = self.graph.add_node(CausalNode {
            label: label.to_string(),
            value,
            confidence: 1.0,
        });
        self.label_to_idx.insert(label.to_string(), idx);
        idx
    }

    /// Add a causal edge: cause → effect.
    pub fn add_edge(&mut self, cause: &str, effect: &str, strength: f32, lag: u32) {
        let cause_idx = self.add_node(cause, 0.0);
        let effect_idx = self.add_node(effect, 0.0);
        self.graph.add_edge(cause_idx, effect_idx, CausalEdge { strength, lag });
    }

    /// Get node by label.
    pub fn get_node(&self, label: &str) -> Option<&CausalNode> {
        self.label_to_idx.get(label)
            .and_then(|&idx| self.graph.node_weight(idx))
    }

    /// List all causes of a node (direct parents).
    pub fn causes(&self, label: &str) -> Vec<(String, f32)> {
        let idx = match self.label_to_idx.get(label) {
            Some(&idx) => idx,
            None => return vec![],
        };
        self.graph.neighbors_directed(idx, petgraph::Incoming)
            .filter_map(|parent| {
                let node = self.graph.node_weight(parent)?;
                let edge = self.graph.edges_connecting(parent, idx).next()?;
                Some((node.label.clone(), edge.weight().strength))
            })
            .collect()
    }

    /// List all effects of a node (direct children).
    pub fn effects(&self, label: &str) -> Vec<(String, f32)> {
        let idx = match self.label_to_idx.get(label) {
            Some(&idx) => idx,
            None => return vec![],
        };
        self.graph.neighbors_directed(idx, petgraph::Outgoing)
            .filter_map(|child| {
                let node = self.graph.node_weight(child)?;
                let edge = self.graph.edges_connecting(idx, child).next()?;
                Some((node.label.clone(), edge.weight().strength))
            })
            .collect()
    }

    /// Check if the graph is a valid DAG (no cycles).
    pub fn is_dag(&self) -> bool {
        !algo::is_cyclic_directed(&self.graph)
    }

    /// Topological sort — returns labels in causal order.
    pub fn topological_order(&self) -> Option<Vec<String>> {
        algo::toposort(&self.graph, None)
            .ok()
            .map(|sorted| {
                sorted.iter()
                    .filter_map(|&idx| self.graph.node_weight(idx).map(|n| n.label.clone()))
                    .collect()
            })
    }

    /// Simulate intervention: set a node's value and propagate downstream.
    ///
    /// For each downstream node: new_value = Σ parent.value × edge.strength
    pub fn intervene(&mut self, label: &str, value: f32) -> Vec<(String, f32)> {
        let sorted = match self.topological_order() {
            Some(s) => s,
            None => return vec![],
        };

        // Set intervention
        if let Some(&idx) = self.label_to_idx.get(label) {
            if let Some(node) = self.graph.node_weight_mut(idx) {
                node.value = value;
            }
        }

        // Propagate in topological order
        let mut changes = vec![];
        let start_pos = sorted.iter().position(|l| l == label).unwrap_or(0);

        for i in (start_pos + 1)..sorted.len() {
            let current_label = &sorted[i];
            let current_idx = match self.label_to_idx.get(current_label) {
                Some(&idx) => idx,
                None => continue,
            };

            // Sum parent contributions
            let mut new_val = 0.0f32;
            let parents: Vec<_> = self.graph
                .neighbors_directed(current_idx, petgraph::Incoming)
                .collect();
            for parent_idx in parents {
                let parent_val = self.graph.node_weight(parent_idx)
                    .map_or(0.0, |n| n.value);
                let edge_strength = self.graph.edges_connecting(parent_idx, current_idx)
                    .next()
                    .map_or(0.0, |e| e.weight().strength);
                new_val += parent_val * edge_strength;
            }

            if let Some(node) = self.graph.node_weight_mut(current_idx) {
                node.value = new_val;
                changes.push((current_label.clone(), new_val));
            }
        }

        changes
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize { self.graph.node_count() }

    /// Number of edges.
    pub fn edge_count(&self) -> usize { self.graph.edge_count() }
}

impl Default for CausalGraph {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_graph() {
        let mut g = CausalGraph::new();
        g.add_edge("rain", "wet_ground", 0.9, 0);
        g.add_edge("wet_ground", "slippery", 0.8, 0);
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
        assert!(g.is_dag());
    }

    #[test]
    fn test_causes_and_effects() {
        let mut g = CausalGraph::new();
        g.add_edge("rain", "wet_ground", 0.9, 0);
        g.add_edge("sprinkler", "wet_ground", 0.7, 0);
        g.add_edge("wet_ground", "slippery", 0.8, 0);

        let causes = g.causes("wet_ground");
        assert_eq!(causes.len(), 2);
        let effects = g.effects("wet_ground");
        assert_eq!(effects.len(), 1);
        assert_eq!(effects[0].0, "slippery");
    }

    #[test]
    fn test_topological_order() {
        let mut g = CausalGraph::new();
        g.add_edge("A", "B", 1.0, 0);
        g.add_edge("B", "C", 1.0, 0);
        let order = g.topological_order().unwrap();
        let pos_a = order.iter().position(|l| l == "A").unwrap();
        let pos_b = order.iter().position(|l| l == "B").unwrap();
        let pos_c = order.iter().position(|l| l == "C").unwrap();
        assert!(pos_a < pos_b && pos_b < pos_c);
    }

    #[test]
    fn test_intervention() {
        let mut g = CausalGraph::new();
        g.add_edge("temp", "pressure", 0.5, 0);
        g.add_edge("pressure", "flow", 0.8, 0);
        let changes = g.intervene("temp", 10.0);
        // pressure = temp * 0.5 = 5.0
        // flow = pressure * 0.8 = 4.0
        assert!(!changes.is_empty());
        let pressure = changes.iter().find(|(l, _)| l == "pressure");
        assert!(pressure.is_some());
        assert!((pressure.unwrap().1 - 5.0).abs() < 0.01);
        let flow = changes.iter().find(|(l, _)| l == "flow");
        assert!(flow.is_some());
        assert!((flow.unwrap().1 - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_add_node_updates() {
        let mut g = CausalGraph::new();
        g.add_node("X", 1.0);
        assert_eq!(g.get_node("X").unwrap().value, 1.0);
        g.add_node("X", 5.0);
        assert_eq!(g.get_node("X").unwrap().value, 5.0);
        assert_eq!(g.node_count(), 1, "Should not duplicate node");
    }

    #[test]
    fn test_empty_causes_effects() {
        let g = CausalGraph::new();
        assert!(g.causes("nonexistent").is_empty());
        assert!(g.effects("nonexistent").is_empty());
    }
}
