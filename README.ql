= Goal =
* Simple shooting game with AI-trained opponent
* Ideally, I could get this running in the browser for super easy sharing

= Todo =
* Basic continuous collision detection
  - could just build this with pymunk instead
* Line of sight implementation
  - idea: can only see things that are not blocked by obstacles
  - need to compute the hidden polygon for all obstacles (some of this can be
    dumped into C++ if needed for speed)
  - idea: if you shoot, you become visible on the map
* Possibly have reloading mechanics as well
