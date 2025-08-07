import networkx as nx
import numpy as np
import pandas
from typing import List, Dict, Any
import re

CELL_PTR = r'cell\( *(\d+), *( *\d+) *\)'

class GlassCell:

    def __init__(self, dag, row, col):
        self.dag = dag
        self.row = row
        self.col = col
        self._template = ''
        self._resolved_template = ''
        self.resolution_count = 0

    @property
    def template(self):
        return self._template
    
    @template.setter
    def template(self, value: str):
        if value != self._template:
            self.invalidate()
            for descendnant in self.get_descendants():
                descendnant.invalidate()
            for ancestor in self.get_ancestors():
                self.dag.remove_edge(ancestor, self)
            self._template = value
            self._resolved_template = self.resolve_template(value)

    def get_ancestors(self) -> List['GlassCell']:
        """ Get all ancestor cells in the DAG."""
        return list(nx.ancestors(self.dag, self))
    
    def get_descendants(self) -> List['GlassCell']:
        """ Get all descendant cells in the DAG.  """
        return list(nx.descendants(self.dag, self))
        
    @property
    def resolved_template(self):
        if not self._resolved_template:
            self._resolved_template = self.resolve_template(self._template)
        return self._resolved_template
    
    def resolve_template(self, template: str) -> str:
        """
        Resolve the template with the provided data.
        This is a placeholder for actual template resolution logic.
        """
        self.resolution_count += 1

        def repl(match):
            row, col = int(match.group(1)), int(match.group(2))
            if not self.dag.has_edge(self.dag.grid[row, col], self):
                self.dag.add_edge(self.dag.grid[row, col], self)
            ref_cell = self.dag.grid[row, col]
            return ref_cell.resolved_template

        resolved = re.sub(CELL_PTR, repl, template)
        return resolved
    
    def invalidate(self):
        self._resolved_template = ''
    
    def __repr__(self):
        return f"<cell({self.row},{self.col})>: {self.template}"
    
    def __hash__(self):
        return hash(f'cell({self.row},{self.col})')
    
class GlassDag(nx.DiGraph):
    """
    A directed acyclic graph (DAG) representing a glass cell structure.
    """

    def __init__(self, rows, cols, *args, **kwargs):
        """Initialize the GlassDag with a grid of GlassCell objects."""
        super().__init__(*args, **kwargs)
        self.grid: np.array[GlassCell] = np.empty((rows, cols), dtype=object)
        self._initialize_grid(rows, cols)

    def _initialize_grid(self, rows: int, cols: int):
        for row in range(rows):
            for col in range(cols):
                self.grid[row,col] = GlassCell(self, row, col)
                self.add_node(self.grid[row, col])

    def resolve_grid(self):
        """
        Resolve all cells in the grid.
        This will trigger the resolution of templates and dependencies.
        """
        for cell in nx.topological_sort(self):
            cell.resolved_template

if __name__ == "__main__":
    # Example usage
    dag = GlassDag(rows=3, cols=3)
    print(dag.grid)  # Should print the initialized grid of GlassCell objects
    cell = dag.grid[1, 0]
    print(cell)  # Should print the representation of the cell at (0, 0)
    
    dag.grid[0,0].template = '0,0'
    dag.grid[0,1].template = '0,1'
    # Example of resolving a template
    cell.template = 'cell(0,0)  + 1'
    resolved = cell.resolved_template
    print(resolved)
    print(dag.has_predecessor(dag.grid[1,0], dag.grid[0,0]))

    cell.template = 'cell(0,1)  + 1'
    resolved = cell.resolved_template
    print(resolved)
    print(dag.has_predecessor(dag.grid[1,0], dag.grid[0,0]))

    import time
    count=30
    dag = GlassDag(count,1)

    dag.grid[0,0].template = '0'
    dag.grid[1,0].template = '1'

    for idx in range(2,count):
        print(f'Adding cell {idx}')
        if idx == 43:
            print(idx)
        dag.grid[idx,0].template = f'cell({idx-2},0) + cell({idx-1},0) + {idx}\n'
        # dag.grid[idx,0].template = f'cell({idx-1},0) + {idx}\n'

    start = time.time()
    print(f'Template: {dag.grid[count-1,0].template}')
    print(f'Resolved template: ')
    dag.grid[count-1,0].resolved_template
    tot_time = time.time() - start
    print(f'Resolution took {tot_time} s')
    print(f'{dag.grid[count-1,0].resolution_count}')
    print(f'{dag.grid[count-10,0].resolution_count}')
