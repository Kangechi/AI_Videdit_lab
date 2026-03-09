Last week I didnt code much, what I did do:
Understanding asset registry and session manager - Joined these two in OOP
- better coding understanding and growth so Today I want to creat a functioning system:
This is because last time when I worked on it 
I didnt necessarily notice a file registrt created.
 
 - One thing I've really gained from last week though is:
 1. File handling - the fact that I can create a json directory to ensure everything is stored and readable
 2. Classes and the conncetion - I remember I made an error during inheritance, so I'm learning and growing
 3. I need to grow in Tkninter and app development as well - and figuring out how all thes things work
 4. Yes

So as I move forward I'm looking forward to keep going



This week:
 Ive really grown alot
 1. Its started with file handling - Being able tocreate a minin database using json and dictionaries
 the use of assets and session - Assets as built in memory(system memory) while sessions as working memory(kinda like Ram)
 - This required OOP - To create the assets class - the methods that aid the assets class 
 class Asset Registry:
  def __init__(self, file= "assets.json"):
      self.file_reg = file
      self.registry = self.load_registry()

 def load_registry(self, file_reg):
 #Load the files that are present in the assets.json
     with open(file_reg, 'r' ) as f:
       return json.load(f)
    return {}
    
def add_asset(self,file_reg ):
 with open(file_reg, "w) as f:
    json.dump(f)

def save_asset(self):


Something like this then for thesession, as it is the working memory it directly connects to the assets

class SessionManager:
 def __init__(self):
     self.session_id = asset.id
 def creat_session(self, asset):


- The challenge with this is that with each run - The data is not able to be retraced by the system again - due to the session which is the current working memory

To solve this - You only have assetregistry and a method that now conncets to the database json plus using cursor AI I learnt a complex means of file handling:

utilising os for mkdir in saving assets then ensuring we use Path(__file__):
-These aid in file handling and preventing the loss of data and also fine tuning the structure of the project

A couple of things I remember is that we also import JSONDecodeError

so:

Yeah and then the intro of GUI -in order to get closer to craeting an MVP of VIDEDIT:
Here we strated by working with Tkinter then finally PySide6 which is better. As PySide6 is prettier

- so adjustong to now working with the GUI and connceting the logic via OOP or separately thats the endavor after my exams.



    
