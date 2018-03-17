import ops

class CelebrityImageInfo:
    def __init__(self, age, identity, year, rank, lfw, birth, name):
        self.age = age
        self.identity = identity
        self.year = year
        self.rank = rank
        self.lfw = lfw
        self.birth = birth
        self.name = name

        self.age_label = ops.age_group_label(self.age)

class CelebrityInfo:
    def __init__(self, name, identity, birth, rank, lfw):
        self.name = name
        self.identity = identity
        self.birth = birth
        self.rank = rank
        self.lfw = lfw

class Images_by_AgeLabel_Name:
    def __init__(self, age_label, name):
        self.age_label = age_label
        self.name = name
        self.dirs = []


