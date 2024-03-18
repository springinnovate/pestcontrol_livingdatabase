VALID_FUNCTIONS = [
    f'{prefix}_{suffix}' for suffix in
    ['mean', 'max', 'min', 'sd']
    for prefix in
    ['spatial', 'years', 'julian']]
print(VALID_FUNCTIONS)

from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

grammar = Grammar(r"""
    function = text "(" args (";" " "* function)? ")"
    args = " "* int ("," " "* int)*
    text = ~"[A-Z_]+"i
    int = ~"(\+|-)?[0-9]*"
    """)

class FunctionProcessor(NodeVisitor):
    def visit_function(self, node, visited_children):
        # Extract the function name and arguments
        function_name, _, args = visited_children[0:3]
        print(f'execute {function_name} with {args}')
        return (function_name, args)

    def visit_args(self, node, visited_children):
        # Process and collect arguments
        first_int = visited_children[1]
        integers = [first_int]
        for _, _, arg in visited_children[2]:
            integers.append(arg)
        return integers

    def visit_text(self, node, visited_children):
        # Return the text directly
        return node.text

    def visit_int(self, node, visited_children):
        # Convert and return the integer
        return int(node.text)

    def generic_visit(self, node, visited_children):
         return visited_children or node


#print(grammar)


'years_mean(-10, 0; julian_sum(220, 2502    , -10, 0; spatial_mean...))'


spatial_mean(
    filter by date,
    aggregate by space,

    point_year, 10000)



function = 'years_mean(-10, 0; seasonal_max(1, 365; spatial_mean(1000)))'
print(function)
tree = grammar.parse(function)
#print(tree)
#print(print(grammar.parse('year_mean(-10, 0, seasonal_max(121, 273))')))
fp = FunctionProcessor()
output = fp.visit(tree)
# for x in output:
#     print(f'line: {x}')


