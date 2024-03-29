
## 如何使用这些函数进行渲染（配合demo.py观看)

<!-- ![demo](../env_data/page.png) -->

### 渲染场景的基本流程
1. 清空场景。
2. 导入模型。
3. 添加相机。
4. 添加光照。
5. 设置渲染参数。
6. 渲染图片。

### 基本模块
`tools`文件夹内提供的几个文件基本对应着上面的步骤：

`scene.py`：对整个场景进行管理，包括清空场景、将物体归类到一个集合（collection）等操作

`models.py`：与模型相关的函数，包括各类网格和点云的加载、几何体的创建

`material.py`：与模型材质相关的函数，主要是在models被调用

`lighter.py`：与光照相关的函数，主要用来加载光源

`viewer.py`：与相机相关的函数，添加相机

`render.py`：与渲染相关的函数，设置渲染参数，进行渲染

`animator.py`：与动画相关的函数，对物体进行关键帧插入

`modifier.py`：与模型处理相关的函数，添加线框、移除重叠网格、光滑表面、组合模型等操作

`video.py`：进行视频生成的函数，将一个图片集合转成视频、视频加速等，具体使用例子在 [`examples/img_to_video.py`](../examples/img_to_video.py)

    !!! 其中一些代码需要将软件设置为英文才能使用，因为有的参数名称会随着默认语言变化而相应变化

### 跑代码前的准备
找到blender文件夹的python文件，我们的代码都是用这个python执行的，参考 `README.md` 的命令安装两个库，trimesh是用来读取点云的，第二个是一个blender的第三方视觉库，可以用来生成实例分割图、深度图等，不过这里安装它只是因为它能帮忙把一些常用的python库也一起安装了（比如numpy和matplotlib）

然后稍微了解一下blender的操作方式

### 简单的使用场景
`demo.py` 中展示了一个创建场景、导入模型到渲染出图的过程，参照自己的需求使用需要的模型加载函数，设置自己需要的相机位置和渲染分辨率。


然后我们可以在不打开blender的情况下，直接用命令行执行：

```blender.exe -b -P demo.py```

你也可以自己在某个地方写一个python文件，调用这里的函数，只要确保 `ROOT_DIR` 这个变量是指向这个函数文件夹就行，但这样没法直接看到自己修改的效果，调整场景参数很麻烦，这个方式最适合的是整体参数设置好，进行批量渲染的时候使用。

设置自己需要的渲染场景时，还是建议打开blender，或者在这里提供的 `demo.blend` 中执行代码（记得先把渲染的函数注释掉，因为还没必要渲染图片），这样可以直接在界面中看到代码执行的效果。具体的blender操作可以参照 `README.md` 中分享的ppt。

自己设置好大致的场景摆放参数后，将这些位置、角度数值写进代码，这时候就可以加个for循环什么的让其自动化执行了。


### 稍微复杂点的使用场景

<details>
<summary>
旋转场景
<p align='center'>
<img src="../doc/images/animate/rot.gif" width="400"></img>
</p>
</summary>

有些时候，我们需要渲染场景的多个视角，直接的想法是我们设置几组相机位置，绕场景一圈拍照渲染就行了，但在光源不动的情况下，这样会得到光照效果不同的几组图，没法用于展示。

如果光源也随着相机旋转，设置会更加麻烦，所以更简单的方式是让场景物体绕着一个Z轴旋转，这样相机和光源都不用修改了。

但我们场景中可能有多个模型组合，为了确保大家都能绕着同一个轴旋转，代码中实现了一个功能：将多个物体设置为一个坐标轴对象的子物体，这样我们旋转平移这个父坐标轴对象的时候，作为子物体的场景都能同时旋转平移。

在blender中我们可以将多个物体设置为同个集合来管理，下面是 `demo.py` 执行后会创建的一些物体，我们从场景列表中可以看到这些物体属于不同的白色盒子图标下，这些白色盒子图标表示的就是不同的collection，

<p align='center'><img src="../doc/images/collection1.png" width="400"></p>

而 [`examples/animation.py`](../examples/animation.py) 中的 `add_scene` 函数，它使用到了一个装饰器 `scene.add_model_in_collection`，这个装饰器的作用是将我们写的函数内添加的物体都放入一个集合collection当中，并创建一个坐标系对象，将这些物体都设置为它的子物体。这个函数执行后得到的结果如下图：

<p align='center'><img src="../doc/images/collection2.png" width="300"></p>


可以看见使用了装饰器来添加物体的话，这些物体不是直接位于集合的第一层，而是作为一个坐标轴对象（Scene_Empty）的子物体加入了场景，这样的好处是我们通过控制这个坐标轴对象就可以让这个场景中所有物体同步位移旋转和缩放。

希望这个装饰器起作用的话，需要在函数的参数内加一个叫做 `collection_name` 的参数，设置为你想要的集合名字。

另外有一点要注意，blender默认设置下渲染的视频可能有些编码问题，会导致没法在Mac上播放，暂时不知道什么原因，所以建议再使用 [handBrake](https://handbrake.fr/) 等视频处理工具再把视频编码一下。

</details>


<details>
<summary>
渲染线框
<p align='center'><img src="../doc/images/wireframe.png"></p>
</summary>

blender内有多种渲染线框的方式，这里实现了3种，分别是位于 `modifier.py` 里面的 `wireframe` 函数，以及 `material.py` 里面的 `wireframe_material` 函数，还有一种基于画笔的方式，位于 `modifier.py` 里面的 `render_with_lines` 函数。在 [`examples/wireframe.py`](../examples/wireframe.py) 中展示了这几个函数的使用方式。


第一种线框方法用的是blender的线框修改器，它会基于当前模型的线框生成一个的线框网格模型，这种方法得到的是一个线框网格mesh。这个修改器可以设置线框使用的材质，这个材质是从物体本身的材质列表中选择的，所以有个offset参数，用于设置材质列表中的第几个材质。

第二种线框方法是从材质的角度，物体的材质中提供了一个叫线框的材质节点，我们可以获取这个信息直接绘制出表面的线段，这种方法是在材质图像层面进行的生成。

第三种画笔方法，也就是正方体例子中最右边的效果，使用的是blender的 `grease pencil`，就是blender里面的2D绘画的笔刷。它可以直接在场景的物体模型上画出我们想要的边缘线段。我们可以用这种方式绘制轮廓线、相交线、网格边。

blender之前想渲染物体轮廓，一般会使用freestyle，这种方法的不方便的地方在于必须渲染后才知道效果。而grease pencil可以实时绘制出这些物体**在相机视角下**的轮廓线，它是独立的物体对象，不会影响其他物体的状态，这种绘制可以同时作用到全局的物体，也可以作用于特定对象，创建方式也很容易，在场景中添加物体选项里面，`Grease Pencil` > `Scene Line Art`。它本质上就是一个空白的grease pencil加上一个 `Line Art` 修改器，所以我们可以在这个修改器的 `Edge Types` 里面选择你想要绘制展示的物体边界信息。

<p align='center'><img src="../doc/images/line_art.jpg" width='400'></p>

这种方式可以方便地得到下面的效果，其中黄色的外轮廓和红色的交界线都是用画笔绘制出来的：

<p align='center'><img src="../doc/images/line_art_example.jpg" width='500'></p>

回到线框的话题，要绘制物体的线框其实还需要额外处理，笔画对象使用的 `Line Art` 修改器中，它的 `Edge Types` 属性有个特别的选项 `Edge Marks`，勾选上后笔画对象可以绘制出进行标记的特定边，标记方法如下：选择我们要处理的网格模型，进入编辑模式，编辑对象切换为 `边模式`，按ctrl+a全选边，右键菜单，有个 `mark freestyle edge`，点击后可以发现物体的边颜色变成了蓝色，这意味着标记成功。退出编辑模式，将笔画对象中修改器的 `Edge Marks` 勾选上，就可以成功把线框绘制出来。

<p align='center'><img src="../doc/images/line_art_mark.jpg" width='500'></p>

标记物体边的操作我们也写成了函数，就在 `modifier.py` 的 `mark_freestyle_edge`，所以这种方法下，完整版的线框渲染包括两步：
标记边，添加笔画，`render_with_lines`中也实现了其他轮廓边的效果，具体可以查看一下函数内容：
```python
modifier.mark_freestyle_edge(obj)
modifier.render_with_lines( 'wireframe', line_thickness, line_color )
```

</details>

<details>
<summary>
模型表面有奇怪黑影，渲染不正常
<p align='center'>
<img src="../doc/images/double.gif" width="400"></img>
</p>
</summary>

如果你看见一个模型里面有些不自然的表面阴影或者一大块黑色，可能有两种原因：

一种是模型表面有多层面片重叠，导致渲染时程序无法确定最终要显示的面片是哪一个，导致区域黑色。在 `env_data/model` 文件夹下提供了一个 `double_face_example.obj`，有兴趣可以导入看看。

这种情况可以先将模型的网格分离成单个面片，再重新拼接，这时候重叠部分会过滤掉。 `modifier.py` 中的 `clean_double_faces` 函数就实现了这个功能。

第二种是模型的法向量有问题，需要重新计算，`modifier.py` 中的 `recalculate_normal` 函数就实现了这个功能，调用了blender的计算操作。但blender的计算方法不能保证一定能修复成功，这时候可以考虑结合上面 `clean_double_faces` 函数，先将模型重组后再计算法向量，这样基本能应对大多数法向量问题。

</details>


<details>
<summary>
模型相交线
<p align='center'><img src="../doc/images/intersect.gif" width="300"></p>
</summary>

使用了blender中的几何节点进行处理，他们在处理模型boolean操作的时候会输出相交边界的信息，我们将其曲线获取后加上厚度就可以得到图上的效果了，具体函数为 `modifier.py` 中的 `show_intersection_lines`。

</details>
