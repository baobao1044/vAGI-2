//! Planning via imagination — forward simulation on causal graph.
//!
//! Given a goal, plan backwards from desired state, simulate forward
//! to check feasibility, score plans by expected outcome.

use crate::causal::CausalGraph;

/// A single planned action.
#[derive(Clone, Debug)]
pub struct PlannedAction {
    /// Which node to intervene on.
    pub target: String,
    /// Value to set.
    pub value: f32,
    /// Expected downstream effects.
    pub expected_effects: Vec<(String, f32)>,
}

/// A complete plan: sequence of actions.
#[derive(Clone, Debug)]
pub struct Plan {
    /// Ordered actions to execute.
    pub actions: Vec<PlannedAction>,
    /// Expected final values of goal nodes.
    pub expected_outcome: Vec<(String, f32)>,
    /// Plan quality score (higher = better).
    pub score: f32,
}

/// Planner that uses the causal graph for forward simulation.
pub struct Planner {
    /// Reference causal graph (will be cloned for simulation).
    graph_template: CausalGraph,
}

impl Planner {
    /// Create planner from a causal graph.
    pub fn new(graph: CausalGraph) -> Self {
        Self { graph_template: graph }
    }

    /// Simulate a single intervention and return downstream effects.
    pub fn simulate_intervention(&self, target: &str, value: f32) -> Vec<(String, f32)> {
        let mut sim = self.graph_template.clone();
        sim.intervene(target, value)
    }

    /// Generate a plan: try each possible intervention, pick the one
    /// that moves goal node closest to desired value.
    ///
    /// Simple greedy planner — not optimal, but demonstrates the pattern.
    pub fn plan_greedy(
        &self,
        goal_node: &str,
        goal_value: f32,
        controllable_nodes: &[&str],
        value_range: (f32, f32),
        n_steps: usize,
    ) -> Plan {
        let mut best_plan = Plan {
            actions: vec![],
            expected_outcome: vec![],
            score: f32::NEG_INFINITY,
        };

        let step_size = (value_range.1 - value_range.0) / n_steps.max(1) as f32;

        for &node in controllable_nodes {
            for step in 0..=n_steps {
                let value = value_range.0 + step as f32 * step_size;
                let effects = self.simulate_intervention(node, value);

                // Find goal node in effects
                let goal_result = effects.iter()
                    .find(|(l, _)| l == goal_node)
                    .map(|(_, v)| *v)
                    .unwrap_or(0.0);

                let error = (goal_result - goal_value).abs();
                let score = -error; // higher = closer to goal

                if score > best_plan.score {
                    best_plan = Plan {
                        actions: vec![PlannedAction {
                            target: node.to_string(),
                            value,
                            expected_effects: effects.clone(),
                        }],
                        expected_outcome: effects,
                        score,
                    };
                }
            }
        }

        best_plan
    }

    /// Access the template graph.
    pub fn graph(&self) -> &CausalGraph { &self.graph_template }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_graph() -> CausalGraph {
        let mut g = CausalGraph::new();
        g.add_edge("temp", "pressure", 0.5, 0);
        g.add_edge("pressure", "flow", 0.8, 0);
        g.add_edge("valve", "flow", 0.3, 0);
        g
    }

    #[test]
    fn test_simulate() {
        let planner = Planner::new(build_test_graph());
        let effects = planner.simulate_intervention("temp", 10.0);
        assert!(!effects.is_empty());
        // pressure = 10 * 0.5 = 5
        // flow = 5 * 0.8 + valve(0) * 0.3 = 4
        let pressure = effects.iter().find(|(l, _)| l == "pressure").unwrap();
        assert!((pressure.1 - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_greedy_plan() {
        let planner = Planner::new(build_test_graph());
        // Goal: flow = 4.0
        // Controllable: temp, valve
        let plan = planner.plan_greedy(
            "flow", 4.0,
            &["temp", "valve"],
            (0.0, 20.0),
            20,
        );
        assert!(!plan.actions.is_empty(), "Should find a plan");
        assert!(plan.score > f32::NEG_INFINITY, "Score should be finite");
        // The plan should achieve flow ≈ 4.0
        let flow_result = plan.expected_outcome.iter()
            .find(|(l, _)| l == "flow")
            .map(|(_, v)| *v);
        assert!(flow_result.is_some(), "Plan should include flow outcome");
        eprintln!("Plan: {:?} → flow={:.2}", plan.actions[0].target, flow_result.unwrap());
    }

    #[test]
    fn test_plan_quality() {
        let planner = Planner::new(build_test_graph());
        let plan = planner.plan_greedy("flow", 4.0, &["temp"], (0.0, 20.0), 100);
        // With fine granularity, should get very close
        let flow = plan.expected_outcome.iter()
            .find(|(l, _)| l == "flow").unwrap().1;
        assert!((flow - 4.0).abs() < 0.5, "Should get close to goal: {flow}");
    }
}
