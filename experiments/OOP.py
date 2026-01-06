class Practice_class:
    def __init__(self,name, age, status):
        self.name = name
        self.age = age
        self.status = status

    def meet(self):
        print("Welcome to the program, please introduce yourself")
        return print(f"I am {self.name}, I am {self.age} years old. I am currently a {self.status}")

    def knowmore(self, hobby, interest, ambition):
        self.hobby = hobby
        self.interest = interest
        self.ambition = ambition
        print("Tell us more about yourself")
        return print(f"I love {self.hobby}, I am interested in {self.interest}, and my ambition is to {self.ambition}")
        

person = Practice_class("Kiboi", 17, "Visionary Leader")
person.meet()
person.knowmore("dancing", ["science", "storytelling","growth"], "Create AG-AI and Co-Creation")
