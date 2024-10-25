= Goal =
* Simple shooting game with AI-trained opponent
* Ideally, I could get this running in the browser for super easy sharing

= Todo =
* Replace pygame with basic collision detection
  - rect/line segment for bullets with obstacles
  - point/line segment distance for players with bullets and obstacles
* Also get rid of numpy interface (this together with removing pymunk
  dependency should allow fairly easy port to JS)
* Health goes down when hit until dead
  - idea: drip blood at low health so you can be tracked
* Reloading
* Idea: if you shoot, you become visible on the map

= Other game =
* Another idea for a game is to have one applying force to some body while
  bouncing off of other bodies subject to gravity and also shooting to propel
  others away (possibly to propel yourself as well?) and seeking to gather
  treasures
* This one could be easier to learn?

= Technical details =
* It would be nice to have a Python interface for learning but then also a
  Javascript interface for easy deployment in the browser
