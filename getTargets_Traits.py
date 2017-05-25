def getTargets_Traits(n):
    if n == 0:
        vecCombTargets = ['JAPANESE POLICE-AGENT', 'JAPANESE DOCTOR',
                          'AUSTRALIAN ACCOUNTANT', 'AUSTRALIAN FIREFIGHTER',
                          "INDIAN POLITICIAN", "INDIAN ACTOR"]
        vecUniTargets = ["POLICE-AGENT", "DOCTOR", "ACCOUNTANT", "FIREFIGHTER", "POLITICIAN", "ACTOR",
                         "JAPANESE", "AUSTRALIAN", "INDIAN"]
        allTargets = vecCombTargets + vecUniTargets
        vecUniTargets = [targ + "_mean" for targ in vecUniTargets]

        TraitsPos = ["Sincere", "Tolerant", "Good-natured", "Trustworthy", "Friendly", "Helpful", "Moral",
                     "Understanding"]
        TraitsNeg = ["Insincere", "Intolerant", "Ill-natured", "Untrustworthy", "Unfriendly", "Unhelpful", "Immoral",
                     "Self-centered"]
        allTraits = TraitsPos + TraitsNeg

    elif n == 1:
        vecCombTargets = ['BRITISH NURSE', 'BRITISH ACCOUNTANT',
                          'ARAB NURSE', 'ARAB ACCOUNTANT']
        vecUniTargets = ["NURSE", "ACCOUNTANT", "BRITISH", "ARAB"]
        allTargets = vecCombTargets + vecUniTargets
        vecUniTargets = [targ + "_mean" for targ in vecUniTargets]

        TraitsPos = ["Sincere", "Tolerant", "Good-natured", "Trustworthy", "Friendly", "Helpful", "Moral",
                     "Understanding"]
        TraitsNeg = ["Insincere", "Intolerant", "Ill-natured", "Untrustworthy", "Unfriendly", "Unhelpful", "Immoral",
                     "Self-centered"]
        allTraits = TraitsPos + TraitsNeg

    elif n == 2:
        vecCombTargets = ['BRITISH NURSE', 'BRITISH ACCOUNTANT', "BRITISH POLITICIAN",
                          'ARAB NURSE', 'ARAB ACCOUNTANT', "ARAB FARMER",
                          "JEWISH FARMER", "JEWISH POLITICIAN", "JEWISH NURSE",
                          "MEXICAN FARMER", "MEXICAN POLITICIAN", "MEXICAN ACCOUNTANT"]
        vecUniTargets = ["NURSE", "ACCOUNTANT", "FARMER", "POLITICIAN", "BRITISH", "ARAB", "JEWISH", "MEXICAN"]
        allTargets = vecCombTargets + vecUniTargets
        vecUniTargets = [targ + "_mean" for targ in vecUniTargets]

        TraitsPos = ["Sincere", "Tolerant", "Good-natured", "Trustworthy", "Friendly", "Helpful", "Moral",
                     "Understanding"]
        TraitsNeg = ["Insincere", "Intolerant", "Ill-natured", "Untrustworthy", "Unfriendly", "Unhelpful", "Immoral",
                     "Self-centered"]
        allTraits = TraitsPos + TraitsNeg
    elif n == 3:
        vecCombTargets = ['LAURIE-ANDERSON NURSE', 'LAURIE-ANDERSON ACCOUNTANT',
                          'EBONY-WILLIAMS NURSE', 'EBONY-WILLIAMS ACCOUNTANT',
                          "DZIELLA-LOUSAPER NURSE", "DZIELLA-LOUSAPER ACCOUNTANT",
                          "MEREDITH-MILLER FARMER", "MEREDITH-MILLER POLITICIAN",
                          "LATOYA-BROWN FARMER", "LATOYA-BROWN POLITICIAN",
                          "AFERDITA-DZAGHIG FARMER", "AFERDITA-DZAGHIG POLITICIAN",
                          ]
        vecUniTargets = ["NURSE", "ACCOUNTANT", "FARMER", "POLITICIAN", "LAURIE-ANDERSON",
                         "EBONY-WILLIAMS", "DZIELLA-LOUSAPER", "MEREDITH-MILLER", "LATOYA-BROWN", "AFERDITA-DZAGHIG"]
        allTargets = vecCombTargets + vecUniTargets
        vecUniTargets = [targ + "_mean" for targ in vecUniTargets]

        TraitsPos = ["Friendly", "Helpful"]
        TraitsNeg = ["Unfriendly", "Unhelpful"]
        allTraits = TraitsPos + TraitsNeg

    return allTargets, allTraits, vecUniTargets