# **⚙️ Project Settings & Configuration**

## **1. System Requirements**

* **OS:** Windows 10/11 or Linux
* **CPU:** Any modern 64-bit processor
* **GPU:** OpenGL **3.3+** compatible graphics card
* **RAM:** 4 GB minimum
* **Tools:**

  * CMake 3.10+
  * g++ / clang / MSVC
  * Git
  * Visual Studio Code or CLion (optional)

---

## **2. Dependencies**

This project uses the following libraries:

* **GLFW** — window, context creation, keyboard input
* **GLAD** — OpenGL function loader
* **TinyGLTF** — for loading `.glb` models
* **GLM** — math library (matrices & vectors)
* **STB Image** — for optional texture loading
* **OpenGL 3.3 Core** — main rendering API

All dependencies are included in the `dependencies/` folder, so **no external installation is needed**.

---

## **3. CMake Configuration**

### **Default build**

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

### **Windows (MSVC)**

```bash
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
```

### **Linux**

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

---

## **4. File Structure**

```
/assets
    whale.glb
    stalactite.glb
/src
    flappy_whale.cpp
    shaders/
        water.vert
        water.frag
        lighting.vert
        lighting.frag
        overlay.vert
        overlay.frag
/dependencies
CMakeLists.txt
README.md
```

---

## **5. Game Settings**

### **Camera**

* Follows the whale smoothly
* Controlled inside code in:
  `camera_position`, `camera_offset`, `glm::lookAt()`

### **Graphics**

* Phong/Blinn–Phong lighting
* Procedural animated water surface
* 3D models scaled automatically
* Depth testing enabled
* Back-face culling enabled

### **Collision**

* Whale: **sphere hitbox**
* Stalactites: **AABB box hitbox**
* Collision formula: sphere–AABB intersection
* Hitbox visualization toggle in debug mode

### **HUD**

* Top-left: **FPS counter**
* Top-center: **Score counter**
* Custom 7-segment renderer (no fonts needed)

---

## **6. Controls**

| Key       | Action                 |
| --------- | ---------------------- |
| **SPACE** | Whale jumps            |
| **R**     | Restart game           |
| **ESC**   | Quit                   |
| **F3**    | Toggle FPS / Debug HUD |

---

## **7. Known Settings You Can Customize**

Inside `flappy_whale.cpp`:

### **Stalactite spacing**

```cpp
const float PIPE_SPACING = 22.0f;
```

### **Gap size**

```cpp
const float GAP_HEIGHT = 6.5f;
```

### **Water speed**

```cpp
uniform float uTime; // automatically updated
```

### **Whale physics**

```cpp
float gravity = -9.8f * 1.7f;
float jumpForce = 7.0f;
```

---

## **8. Running the Game**

```bash
./FlappyWhale3D
```

The game runs in **fullscreen mode** by default.
Change to windowed mode in `glfwCreateWindow()` settings.

---

## **9. Troubleshooting**

### **Black screen**

* Ensure your GPU supports OpenGL 3.3 Core.
* Update graphics drivers.

### **Models not loading**

* Check `assets/` folder path.
* Ensure `whale.glb` and `stalactite.glb` exist.

### **Shaders failing**

Use:

```cpp
glGetShaderInfoLog()
glGetProgramInfoLog()
```

---

