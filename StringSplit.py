def StringSplit(Key):
    """ This function separetes the key from the pandas DataFrame into separate sections which are relevant
    Dependent upon input -->
    label[1] = combined target
    label[2] = Ethnicity /Name
    label[3] = occupation
    """
    label, label_title = {}, {}

    label[1] = Key
    [label_title[1], endmark] = label[1].split("_")
    [label_title[2], label[3]] = label[1].split(" ")
    [label_title[3], endmark] = label[3].split("_")
    label[2] = label_title[2]+"_"+endmark
    # import pdb
    # pdb.set_trace()

    return label, label_title