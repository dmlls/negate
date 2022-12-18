"""Data for tests.

Format::

   (input_sentence, output_sentence, prefer_contractions)
"""

# AUX as ROOT - Affirmative
aux_root_affirmative = [
    ("I should.", "I shouldn't.", True),
    ("I should.", "I should not.", False),
    ("I can.", "I can't.", True),
    ("I can.", "I cannot.", False),
    ("I will.", "I won't.", True),
    ("I will.", "I will not.", False),
    ("I'm excited.", "I'm not excited.", True),
    ("I'm excited.", "I am not excited.", False),
    ("I am excited.", "I am not excited.", True),
    ("I am excited.", "I am not excited.", False),
    ("I've been excited.", "I haven't been excited.", True),
    ("I've been excited.", "I have not been excited.", False),
    ("I have been excited.", "I haven't been excited.", True),
    ("I have been excited.", "I have not been excited.", False),
    ("I'll be excited.", "I won't be excited.", True),
    ("I'll be excited.", "I will not be excited.", False),
    ("I will be excited.", "I won't be excited.", True),
    ("I will be excited.", "I will not be excited.", False),
    ("She's excited.", "She isn't excited.", True),
    ("She's excited.", "She is not excited.", False),
    ("She is excited.", "She isn't excited.", True),
    ("She is excited.", "She is not excited.", False),
    ("She's been excited.", "She hasn't been excited.", True),
    ("She's been excited.", "She has not been excited.", False),
    ("She has been excited.", "She hasn't been excited.", True),
    ("She has been excited.", "She has not been excited.", False),
    ("You're excited.", "You aren't excited.", True),
    ("You're excited.", "You are not excited.", False),
    ("You are excited.", "You aren't excited.", True),
    ("You are excited.", "You are not excited.", False),
    ("We're excited.", "We aren't excited.", True),
    ("We're excited.", "We are not excited.", False),
    ("We are excited.", "We aren't excited.", True),
    ("We are excited.", "We are not excited.", False),
    ("They're excited.", "They aren't excited.", True),
    ("They're excited.", "They are not excited.", False),
    ("They are excited.", "They aren't excited.", True),
    ("They are excited.", "They are not excited.", False),
]

# AUX as ROOT - Negative
aux_root_negative = [
    ("I don't.", "I do.", True),
    ("I don't.", "I do.", False),
    ("I do not.", "I do.", True),
    ("I do not.", "I do.", False),
    ("She doesn't.", "She does.", True),
    ("She doesn't.", "She does.", False),
    ("She does not.", "She does.", True),
    ("She does not.", "She does.", False),
    ("She doesn't.", "She does.", True),
    ("She doesn't.", "She does.", False),
    ("She does not.", "She does.", True),
    ("She does not.", "She does.", False),
    ("I shouldn't.", "I should.", True),
    ("I shouldn't.", "I should.", False),
    ("I should not.", "I should.", True),
    ("I should not.", "I should.", False),
    ("I can't.", "I can.", True),
    ("I can't.", "I can.", False),
    ("I cannot.", "I can.", True),
    ("I cannot.", "I can.", False),
    ("I won't.", "I will.", True),
    ("I won't.", "I will.", False),
    ("I will not.", "I will.", True),
    ("I will not.", "I will.", False),
    ("I'm not excited.", "I'm excited.", True),
    ("I'm not excited.", "I'm excited.", False),
    ("I am not excited.", "I am excited.", True),
    ("I am not excited.", "I am excited.", False),
    ("I've not been excited.", "I've been excited.", True),
    ("I've not been excited.", "I've been excited.", False),
    ("I have not been excited.", "I have been excited.", True),
    ("I have not been excited.", "I have been excited.", False),
    ("I'll not be excited.", "I'll be excited.", True),
    ("I'll not be excited.", "I'll be excited.", False),
    ("I won't be excited.", "I will be excited.", True),
    ("I won't be excited.", "I will be excited.", False),
    ("I will not be excited.", "I will be excited.", True),
    ("I will not be excited.", "I will be excited.", False),
    ("He's not excited.", "He's excited.", True),
    ("He's not excited.", "He's excited.", False),
    ("He is not excited.", "He is excited.", True),
    ("He is not excited.", "He is excited.", False),
    ("We're not excited.", "We're excited.", True),
    ("We're not excited.", "We're excited.", False),
    ("We are not excited.", "We are excited.", True),
    ("We are not excited.", "We are excited.", False),
    ("You're not excited.", "You're excited.", True),
    ("You're not excited.", "You're excited.", False),
    ("You are not excited.", "You are excited.", True),
    ("You are not excited.", "You are excited.", False),
    ("They're not excited.", "They're excited.", True),
    ("They're not excited.", "They're excited.", False),
    ("They are not excited.", "They are excited.", True),
    ("They are not excited.", "They are excited.", False),
]

# AUX in ROOT children - Affirmative
aux_root_children_affirmative = [
    ("I do think it's true.", "I don't think it's true.", True),
    ("I do think it's true.", "I do not think it's true.", False),
    ("She does think it's true.", "She doesn't think it's true.", True),
    ("She does think it's true.", "She does not think it's true.", False),
    ("I should do it.", "I shouldn't do it.", True),
    ("I should do it.", "I should not do it.", False),
    ("You ought to do it.", "You ought not to do it.", False),
    ("I can do it.", "I can't do it.", True),
    ("I can do it.", "I cannot do it.", False),
    ("I will do it.", "I won't do it.", True),
    ("I will do it.", "I will not do it.", False),
    ("I've done it.", "I haven't done it.", True),
    ("I've done it.", "I have not done it.", False),
    ("I've been doing it.", "I haven't been doing it.", True),
    ("I've been doing it.", "I have not been doing it.", False),
    ("I have done it.", "I haven't done it.", True),
    ("I have done it.", "I have not done it.", False),
    ("I have been doing it.", "I haven't been doing it.", True),
    ("I have been doing it.", "I have not been doing it.", False),
    ("I'd done it.", "I hadn't done it.", True),
    ("I'd done it.", "I had not done it.", False),
    ("I had done it.", "I hadn't done it.", True),
    ("I had done it.", "I had not done it.", False),
    ("I had been doing it.", "I hadn't been doing it.", True),
    ("I had been doing it.", "I had not been doing it.", False),
    ("She's done it.", "She hasn't done it.", True),
    ("She's done it.", "She has not done it.", False),
    ("She's been doing it.", "She hasn't been doing it.", True),
    ("She's been doing it.", "She has not been doing it.", False),
    ("She has done it.", "She hasn't done it.", True),
    ("She has done it.", "She has not done it.", False),
    ("She has been doing it.", "She hasn't been doing it.", True),
    ("She has been doing it.", "She has not been doing it.", False),
    ("She'd done it.", "She hadn't done it.", True),
    ("She'd done it.", "She had not done it.", False),
    ("She'd been doing it.", "She hadn't been doing it.", True),
    ("She'd been doing it.", "She had not been doing it.", False),
    ("She had done it.", "She hadn't done it.", True),
    ("She had done it.", "She had not done it.", False),
    ("She had been doing it.", "She hadn't been doing it.", True),
    ("She had been doing it.", "She had not been doing it.", False),
    ("I would have done it.", "I wouldn't have done it.", True),
    ("I would have done it.", "I would not have done it.", False),
    ("I would've done it.", "I wouldn't have done it.", True),
    ("I would've done it.", "I would not have done it.", False),
    ("I should have done it.", "I shouldn't have done it.", True),
    ("I should have done it.", "I should not have done it.", False),
    ("I should've done it.", "I shouldn't have done it.", True),
    ("I should've done it.", "I should not have done it.", False),
]

# AUX in ROOT children - Negative
aux_root_children_negative = [
    ("I shouldn't do it.", "I should do it.", True),
    ("I shouldn't do it.", "I should do it.", False),
    ("I should not do it.", "I should do it.", True),
    ("I should not do it.", "I should do it.", False),
    ("I can't do it.", "I can do it.", True),
    ("I can't do it.", "I can do it.", False),
    ("I cannot do it.", "I can do it.", True),
    ("I cannot do it.", "I can do it.", False),
    ("I won't do it.", "I will do it.", True),
    ("I won't do it.", "I will do it.", False),
    ("I will not do it.", "I will do it.", True),
    ("I will no do it.", "I will do it.", False),
    ("I've not done it.", "I've done it.", True),
    ("I've not done it.", "I've done it.", False),
    ("I've not been doing it.", "I've been doing it.", True),
    ("I've not been doing it.", "I've been doing it.", False),
    ("I haven't done it.", "I have done it.", True),
    ("I haven't done it.", "I have done it.", False),
    ("I haven't been doing it.", "I have been doing it.", True),
    ("I haven't been doing it.", "I have been doing it.", False),
    ("I have not done it.", "I have done it.", True),
    ("I have not done it.", "I have done it.", False),
    ("I have not been doing it.", "I have been doing it.", True),
    ("I have not been doing it.", "I have been doing it.", False),
    ("I'd not done it.", "I'd done it.", True),
    ("I'd not done it.", "I'd done it.", False),
    ("I'd not been doing it.", "I'd been doing it.", True),
    ("I'd not been doing it.", "I'd been doing it.", False),
    ("I hadn't done it.", "I had done it.", True),
    ("I hadn't done it.", "I had done it.", False),
    ("I hadn't been doing it.", "I had been doing it.", True),
    ("I hadn't been doing it.", "I had been doing it.", False),
    ("I had not done it.", "I had done it.", True),
    ("I had not done it.", "I had done it.", False),
    ("I had not been doing it.", "I had been doing it.", True),
    ("I had not been doing it.", "I had been doing it.", False),
    ("She's not done it.", "She's done it.", True),
    ("She's not done it.", "She's done it.", False),
    ("She's not been doing it.", "She's been doing it.", True),
    ("She's not been doing it.", "She's been doing it.", False),
    ("She hasn't done it.", "She has done it.", True),
    ("She hasn't done it.", "She has done it.", False),
    ("She hasn't been doing it.", "She has been doing it.", True),
    ("She hasn't been doing it.", "She has been doing it.", False),
    ("She has not done it.", "She has done it.", True),
    ("She has not done it.", "She has done it.", False),
    ("She has not been doing it.", "She has been doing it.", True),
    ("She has not been doing it.", "She has been doing it.", False),
    ("I wouldn't have done it.", "I would have done it.", True),
    ("I wouldn't have done it.", "I would have done it.", False),
    ("I would not have done it.", "I would have done it.", True),
    ("I would not have done it.", "I would have done it.", False),
    ("I shouldn't have done it.", "I should have done it.", True),
    ("I shouldn't have done it.", "I should have done it.", False),
    ("I should not have done it.", "I should have done it.", True),
    ("I should not have done it.", "I should have done it.", False),
]

# General verbs - Affirmative
general_verbs_affirmative = [
    ("I love hiking.", "I don't love hiking.", True),
    ("I love hiking.", "I do not love hiking.", False),
    ("He loves hiking.", "He doesn't love hiking.", True),
    ("He loves hiking.", "He does not love hiking.", False),
    ("I used to love hiking.", "I didn't use to love hiking.", True),
    ("I used to love hiking.", "I did not use to love hiking.", False),
    ("I really liked the food.", "I really didn't like the food.", True),
    ("I really liked the food.", "I really did not like the food.", False),
]

# General verbs - Negative
general_verbs_negative = [
    ("I don't think it's true.", "I think it's true.", True),
    ("I don't think it's true.", "I think it's true.", False),
    ("I do not think it's true.", "I think it's true.", True),
    ("I do not think it's true.", "I think it's true.", False),
    ("She doesn't think it's true.", "She thinks it's true.", True),
    ("She doesn't think it's true.", "She thinks it's true.", False),
    ("She does not think it's true.", "She thinks it's true.", True),
    ("She does not think it's true.", "She thinks it's true.", False),
]

# Miscellaneous
misc = [
    ("A small Python module that doesn't negate sentences.", "A small Python module that negates sentences.", True),
    ("You should always be careful.", "You shouldn't always be careful.", True),
    ("You must be careful.", "You must not be careful.", False),
    ("I would have done it differently.", "I wouldn't have done it differently.", True),
    ("They are thinking too much and things might not go right.", "They aren't thinking too much and things might not go right.", True),
    ("They aren't going too fast and they might not have time.", "They are going too fast and they might not have time.", True),
    ("I am tired.", "I am not tired.", False),
    ("This was incorrectly assumed.", "This was not incorrectly assumed.", False),
    ("They were brought up in a small town.", "They weren't brought up in a small town.", True),
    ("I do like it.", "I don't like it.", True),
    ("She does like it.", "She doesn't like it.", True),
    ("He does like it.", "He does not like it.", False),
    ("He likes it.", "He doesn't like it.", True),
    ("He doesn't like it.", "He likes it.", True),
    ("I know everything.", "I don't know everything.", True),
    ("I don't know everything.", "I know everything.", True),
    ("The latter's chariot's wheels sank into the ground.", "The latter's chariot's wheels didn't sink into the ground.", True),
    ("He prepared a delicious meal.", "He didn't prepare a delicious meal.", True),
    ("I used to know everything.", "I did not use to know everything.", False),
    ("I'm so damn hungry.", "I'm not so damn hungry.", True),
    ("I am so damn hungry.", "I am not so damn hungry.", True),
    ("I am so damn hungry.", "I am not so damn hungry.", False),
    ("He's so afraid.", "He isn't so afraid.", True),
    ("He's so afraid.", "He is not so afraid.", False),
    ("They are kinda crazy.", "They are not kinda crazy.", False),
    ("There are many ways it can be done.", "There are not many ways it can be done.", False),
    ("There are many ways it can be done.", "There aren't many ways it can be done.", True),
    ("There exist many ways it can be done.", "There don't exist many ways it can be done.", True),
    ("For more info, visit our webpage.", "For more info, don't visit our webpage.", True),
    ("To dream, one doesn't need to sleep.", "To dream, one needs to sleep.", True),
    ("Make an iterator that returns object over and over again.", "Do not make an iterator that returns object over and over again.", False),
    ("Make an iterator that returns object over and over again.", "Don't make an iterator that returns object over and over again.", True),
    ("Do that again.", "Do not do that again.", False),
    ("Do that again.", "Don't do that again.", True),
    ("They will do it asap.", "They won't do it asap.", True),
    ("Not that this will ever change.", "Not that this won't ever change.", True),
    ("Not that this will ever change.", "Not that this will not ever change.", False),
    ("Not that this will not ever change.", "Not that this will ever change.", False),
    ("Not that this won't ever change.", "Not that this will ever change.", False),
    ("Not that he thinks a lot.", "Not that he doesn't think a lot.", True),
    ("Not that he thinks a lot.", "Not that he does not think a lot.", False),
    ("Not that he doesn't think a lot.", "Not that he thinks a lot.", False),
    ("Not that he does not think a lot.", "Not that he thinks a lot.", False),
    ("Not for the first time, she felt utterly betrayed.", "Not for the first time, she didn't feel utterly betrayed.", True),
    ("I've studied hard.", "I haven't studied hard.", True),
    ("I've studied hard.", "I have not studied hard.", False),
    ("She's been to Paris.", "She has not been to Paris.", False),
    ("She's done it.", "She hasn't done it.", True),
    ("She's running.", "She isn't running.", True),
    ("He's rather shy.", "He isn't rather shy.", True),
    ("He's Paul.", "He is not Paul.", False),
    ("I will never go there.", "I will go there.", False),
    ("I always go there.", "I always don't go there.", True)  # not very natural
]

# Sentences that are currently failing.
failing = [
    # POS tagger misclassifications.
    ("I do.", "I don't.", True),  # "do" classified as VERB instead of AUX.
    ("I do.", "I do not.", False),  # "do" classified as VERB instead of AUX.
    ("She does.", "She doesn't.", True),  # "do" classified as VERB instead of AUX.
    ("She does.", "She does not.", False),  # "do" classified as VERB instead of AUX.
    ("She's determined.", "She is not determined.", False),  # "determined" classified as VERB instead of ADJ.
    # Connectors.
    ("It also prohibits or restricts the use of certain mechanisms.", "It also doesn't prohibit or restrict the use of certain mechanisms.", True),
    # Boolean logic, e.g.: "prohibits AND restricts" -> "doesn't prohibit OR restrict"
    ("It also prohibits and restricts the use of certain mechanisms.", "It also doesn't prohibit or restrict the use of certain mechanisms.", True),
    # Non-verbal negations.
    ("A piece with no moving parts.", "A piece with moving parts.", False),  # No verb to negate.
    # Special case of negated ought not yet implemented.
    ("You ought not to do it.", "You ought to do it.", False),
    ("You ought to do it.", "You oughtn't do it.", True),
    ("You oughtn't do it.", "You ought to do it.", True),
]