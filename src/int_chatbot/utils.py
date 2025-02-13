def format_results(columns: list[str], results: list) -> list:
    """
    Format the results of a SQL query into a list of dictionaries.

    Args:
        columns: list[str], the column names
        results: list, the results of the query

    Returns:
        list: the formatted results
    """
    output = []
    for row in results:
        formatted_row = dict(zip(columns, row))
        output.append(formatted_row)

    return output
