def tuple_to_dict(name_list, dict):
        new_dict = {}
        ss = len(name_list)
        for key, value_tuple in dict.items():
            for i in range(ss):
                new_key = key+name_list[i]  
                new_dict[new_key] = float(value_tuple[i])

