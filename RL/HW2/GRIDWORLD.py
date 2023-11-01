import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def change_range(values, vmin=0, vmax=1):
    start_zero = values - np.min(values)
    return (start_zero / (np.max(start_zero) + 1e-7)) * (vmax - vmin) + vmin

class GridWorld:
    # 定义不同地形的颜色
    terrain_color = dict(
        normal=[127 / 360, 0, 96 / 100],
        objective=[26 / 360, 100 / 100, 100 / 100],
        cliff=[247 / 360, 92 / 100, 70 / 100],
        player=[344 / 360, 93 / 100, 100 / 100],
    )

    def __init__(self):
        # 初始化GridWorld对象
        self.player = None
        self._create_grid()  # 创建网格
        self._draw_grid()  # 绘制网格
        self.num_steps = 0  # 步数计数器

    def _create_grid(self, initial_grid=None):
        # 创建网格，初始化为普通地形颜色
        self.grid = self.terrain_color["normal"] * np.ones((4, 12, 3))
        self._add_objectives(self.grid)  # 在网格中添加目标和悬崖

    def _add_objectives(self, grid):
        # 在网格中指定位置添加目标和悬崖颜色
        grid[-1, 1:11] = self.terrain_color["cliff"]
        grid[-1, -1] = self.terrain_color["objective"]

    def _draw_grid(self):
        # 绘制网格界面
        self.fig, self.ax = plt.subplots(figsize=(12, 4))
        self.ax.grid(which="minor")
        self.q_texts = [
            self.ax.text(
                *self._id_to_position(i)[::-1],
                "0",
                fontsize=11,
                verticalalignment="center",
                horizontalalignment="center"
            )
            for i in range(12 * 4)
        ]

        self.im = self.ax.imshow(
            hsv_to_rgb(self.grid),
            cmap="terrain",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        self.ax.set_xticks(np.arange(12))
        self.ax.set_xticks(np.arange(12) - 0.5, minor=True)
        self.ax.set_yticks(np.arange(4))
        self.ax.set_yticks(np.arange(4) - 0.5, minor=True)

    def reset(self):
        # 重置环境，将智能体放置在初始位置，并返回初始状态的ID
        self.player = (3, 0)
        self.num_steps = 0
        return self._position_to_id(self.player)

    def step(self, action):
        # 执行一个动作，更新智能体的位置，并返回状态、奖励和是否完成的信息
        if action == 0 and self.player[0] > 0:
            self.player = (self.player[0] - 1, self.player[1])
        if action == 1 and self.player[0] < 3:
            self.player = (self.player[0] + 1, self.player[1])
        if action == 2 and self.player[1] < 11:
            self.player = (self.player[0], self.player[1] + 1)
        if action == 3 and self.player[1] > 0:
            self.player = (self.player[0], self.player[1] - 1)

        self.num_steps = self.num_steps + 1

        if all(self.grid[self.player] == self.terrain_color["cliff"]):
            reward = -100
            done = True
        elif all(self.grid[self.player] == self.terrain_color["objective"]):
            reward = 0
            done = True
        else:
            reward = -1
            done = False

        return self._position_to_id(self.player), reward, done

    def _position_to_id(self, pos):
        """将位置坐标映射到唯一的ID"""
        return pos[0] * 12 + pos[1]

    def _id_to_position(self, idx):
        return (idx // 12), (idx % 12)

    def render(self, q_values=None, action=None, max_q=False, colorize_q=False):
        assert self.player is not None, "首先需要调用.reset()"

        if colorize_q:
            assert q_values is not None, "使用colorize_q时q_values不能为None"
            grid = self.terrain_color["normal"] * np.ones((4, 12, 3))
            values = change_range(np.max(q_values, -1)).reshape(4, 12)
            grid[:, :, 1] = values
            self._add_objectives(grid)
        else:
            grid = self.grid.copy()

        grid[self.player] = self.terrain_color["player"]
        self.im.set_data(hsv_to_rgb(grid))

        if q_values is not None:
            xs = np.repeat(np.arange(12), 4)
            ys = np.tile(np.arange(4), 12)

            for i, text in enumerate(self.q_texts):
                if max_q:
                    q = max(q_values[i])
                    txt = "{:.2f}".format(q)
                    text.set_text(txt)
                else:
                    actions = ["U", "D", "R", "L"]
                    txt = "\n".join(
                        [
                            "{}: {:.2f}".format(k, q)
                            for k, q in zip(actions, q_values[i])
                        ]
                    )
                    text.set_text(txt)

        if action is not None:
            self.ax.set_title(action, color="r", weight="bold", fontsize=32)

        plt.pause(0.5)