The two weeks I was offline introduced me to Sytems's thinking and its role in how we write code and manipulate the system were creating with our code.
I've come to  understand:
   Nodes- functions: These perform one task each.
   -> They work in contracts of the data they take(data shape)-> the function they perform and in each function there has to be validation, normalization, 
   -> After that, data output shape - the return shape which maps onto the downstream function
An example here is:
   def load_video(video_path: str) -> dict:
      if not isinstance():
       raise valueError("Not string")
      if not os.path.exist():
        raise ValueError("File unable to open")

        clip = cv.VideoCapture(video_path)
        if not clip.isOpened():
        raise valueError()

        extract metadata etc
         return {
            dict
         }
 

Basically the contracts in the first function, map onto the downstream functioning
You also have to think about error and failure raising and surfacing.

Then all these map onto how you cleanly write code:
As I have to intuintively think about these things as I move forward with Videdit
A system to edit videos, that applies co-relation and co-working with an AI/Automatic system to achieve the desired edit.
The user is in control while the system is an aid and does the heacy work - also perhaps the integration of the instagram feature of making collage in stories
