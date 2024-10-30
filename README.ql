= Goal =
* Simple shooting game with AI-trained opponent
* Ideally, I could get this running in the browser for super easy sharing

= Todo =
* Also get rid of numpy interface (this together with removing pymunk
  dependency should allow fairly easy port to JS)
* Idea: if you shoot, you become visible on the map
* Idea: tag. When it, goal is to catch the other player. Key is to tune the
  speeds:
  - backward: slow
  - forward (it): medium
  - forward (not it): fast
* This would allow me to avoid projectiles, ammo, and health mechanics, though
  I could bring in shooting type powerups if needed.

= Other game =
* Another idea for a game is to have one applying force to some body while
  bouncing off of other bodies subject to gravity and also shooting to propel
  others away (possibly to propel yourself as well?) and seeking to gather
  treasures
* This one could be easier to learn?

= Technical details =
* It would be nice to have a Python interface for learning but then also a
  Javascript interface for easy deployment in the browser

= Learning =
* Action space: directions of motion, reload?, target
* Observation space: position of all agents? but also need to encode
  obstacles... this may require learning from pixels directly!
