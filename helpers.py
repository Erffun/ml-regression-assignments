def get_lowest_index(input_list:list)->dict:
    list_dict = dict()
    indx = 0

    for rss in input_list:
        list_dict[indx] = rss
        indx += 1

    min = min(list_dict, key=list_dict.get)
    return min