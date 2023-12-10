def get_file_names(start, end, data_path)->list:
    """Gets the file names between the start and end. 
    Example parameter: 20150101 is January 1st 2015. yyyymmdd
    
    Args: 
    start (string): Start date of training
    end (string): Past the end date of training. Not included

    Returns: 
        List[string]: List of all training files. 
    """
    files = os.listdir(data_path)
    files = sorted(filter(lambda fname: fname < f"data.{end}" and fname >= f"data.{start}", files))
    return files

def filter_df():
    None