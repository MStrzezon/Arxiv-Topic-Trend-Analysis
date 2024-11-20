import ast


def parse_list_column(column):
    return ast.literal_eval(column)