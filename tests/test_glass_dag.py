import unittest
import time
from glass_dag.dag_model import GlassDag, GlassCell

class TestGlassDag(unittest.TestCase):
    def test_grid_initialization(self):
        dag = GlassDag(2, 2)
        self.assertEqual(dag.grid.shape, (2, 2))
        self.assertIsInstance(dag.grid[0, 0], GlassCell)

    def test_template_assignment_and_resolution(self):
        dag = GlassDag(2, 2)
        dag.grid[0, 0].template = '5'
        dag.grid[0, 1].template = 'cell(0,0) + 1'
        self.assertEqual(dag.grid[0, 0].resolved_template, '5')
        self.assertIn('5', dag.grid[0, 1].resolved_template)

    def test_dependency_graph(self):
        dag = GlassDag(2, 2)
        dag.grid[0, 0].template = '5'
        dag.grid[0, 1].template = 'cell(0,0) + 1'
        self.assertTrue(dag.has_predecessor(dag.grid[0, 1], dag.grid[0, 0]))
        dag.grid[0, 1].template = '7'
        self.assertFalse(dag.has_predecessor(dag.grid[0, 1], dag.grid[0, 0]))

    def test_resolve_grid(self):
        dag = GlassDag(3, 1)
        dag.grid[0, 0].template = 'ABC'
        dag.grid[1, 0].template = 'cell(0,0) + 1'
        dag.grid[2, 0].template = 'cell(1,0) + 1'
        dag.resolve_grid()
        self.assertEqual(dag.grid[0, 0].resolved_template, 'ABC')
        self.assertIn('ABC', dag.grid[1, 0].resolved_template)
        self.assertIn(dag.grid[1, 0].resolved_template, dag.grid[2, 0].resolved_template)


class TestGlassDagPerformance(unittest.TestCase):
    def test_long_chain_performance(self):
        count = 1000
        dag = GlassDag(count, 1)
        dag.grid[0, 0].template = '0'
        dag.grid[1, 0].template = 'cell(0,0) + 1'
        for idx in range(2, count):
            dag.grid[idx, 0].template = f'cell({idx-1},0) + {idx}'
        start = time.time()
        resolved = dag.grid[count-1, 0].resolved_template
        elapsed = time.time() - start
        print(f"Resolved {count} cells in {elapsed:.4f} seconds")
        self.assertIsInstance(resolved, str)
        self.assertLess(elapsed, 2.0)  # Should resolve within 2 seconds for 1000 cells

    def test_double_dependency_performance(self):
        count = 40
        dag = GlassDag(count, 1)
        dag.grid[0, 0].template = '0'
        dag.grid[1, 0].template = '1'
        for idx in range(2, count):
            dag.grid[idx, 0].template = f'cell({idx-1},0) + cell({idx-2},0)'
        start = time.time()
        resolved = dag.grid[count-1, 0].resolved_template
        elapsed = time.time() - start
        print(f"Resolved {count} cells with double dependencies in {elapsed:.4f} seconds")
        self.assertIsInstance(resolved, str)
        self.assertLess(elapsed, 2.0)  # Should resolve within 2 seconds for 1000 cells with double dependencies

if __name__ == '__main__':
    unittest.main()
