# Plot

import matplotlib.pyplot as plt

### line

---

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/Python/images/line.png" width="100%" /><br/>
</div>

``` python
import numpy as np
import matplotlib.pyplot as plt

NUM_MODELS = 5
TASKS = ['FloorPlan1#5', 'FloorPlan2#13', 'FloorPlan3#8', 'FloorPlan4#248', 'FloorPlan5#31', 'FloorPlan6#62',
         'FloorPlan7#141', 'FloorPlan8#256', 'FloorPlan9#52', 'FloorPlan10#46', 'FloorPlan201#432']
MODELS = ['1', '2', '3', '4', '5']
MARKES = ['-o', '-s', '-^', '-p', '-v']


def result_plot(data):
    plt.figure(figsize=(10, 5))
    plt.xlabel('scene')
    plt.ylabel('reward')

    for i in range(NUM_MODELS):
        x = []
        y = []
        for scene in TASKS:
            scene_scope, task_scope = scene.split('#')
            for key, value in data[i].items():
                sub_scene, sub_task = key.split('#')
                if sub_scene == scene_scope and sub_task == task_scope:
                    if sub_scene[-3].isdigit():
                        scene_num = sub_scene[-3:]
                    elif sub_scene[-2].isdigit():
                        scene_num = sub_scene[-2:]
                    else:
                        scene_num = sub_scene[-1]
                    x_label = 'FP' + scene_num + '#' + sub_task
                    x.append(x_label)
                    y.append(value)
        plt.plot(x, y, MARKES[i])
        plt.xticks(rotation=270)

    plt.legend(MODELS, loc='upper right')
    plt.tight_layout()
    plt.show()

    # plt.savefig('./reward.png')
    # plt.close()


if __name__ == '__main__':
    scene_states = dict()

    for i in range(NUM_MODELS):
        scene_states[i] = dict()
        for scene in TASKS:
            data = np.random.randint(0, 20, size=100)
            scene_states[i][scene] = data

    rewards = dict()

    for i in range(NUM_MODELS):
        rewards[i] = dict()
        for scene in TASKS:
            rewards[i][scene] = np.mean(scene_states[i][scene])

    result_plot(rewards)
```
