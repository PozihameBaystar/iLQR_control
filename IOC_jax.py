import jax
import jax.numpy as jnp
import numpy as np
import math
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt


#離散化状態方程式
@jax.jit
def model_func_risan(x, u, dt):
    del_x = jnp.array([(x[2]+u[0])*dt, (x[3]+u[1])*dt, u[0], u[1]])
    x_next = x + del_x
    return x_next

model_dfdx = jax.jit(jax.jacfwd(model_func_risan,0))
model_dfdu = jax.jit(jax.jacfwd(model_func_risan,1))


## パラメーター
@dataclass
class ioc_args:
    # 評価関数関連
    S = jnp.eye(2)
    Q = jnp.eye(2)  
    R = jnp.eye(2)  # 速度の重み
    r = jnp.eye(2)  # 速度変化の重み
    D = 1.0  # 目的方向への余弦の重み
    E = 1.0  # 回避関数の重み
    x_ob = jnp.array([[5,4]])
    x_ev = jnp.array([[3,2]])
    alpha = 1.0 # 逆最適制御の尖り具合
    sigma = 1.0 # カーネル関数の分散
    kyori = [0.45, 1.2, 3.6]
    weights = jnp.ones((3,6))
    obs_dim = 4
    action_dim = 2
    # コントローラーのその他のパラメータ
    Ts = 0.1
    tf = 1.0
    N = int(tf/Ts)
    dt = Ts

ioc_args = ioc_args()





# 尤度関数
@jax.jit
def IOC_yuudo(xs,us,S,Q,R,r,D,E,weights):

    # カーネル関数
    def kernel(x,kyori):
        return jnp.exp( -1/(2*ioc_args.sigma**2) * (jnp.linalg.norm(x-ioc_args.x_ev)-kyori)**2 )
    
    def weight_kernel(x,weights):
        return 1 + weights[0]*kernel(x,ioc_args.kyori[0]) + weights[1]*kernel(x,ioc_args.kyori[1]) + weights[2]*kernel(x,ioc_args.kyori[2])

    # ステージコスト
    def stage_cost(x,u,weights):
        cost = 0.5 * ( weight_kernel(x[:2],weights[0]) * (x[:2]-ioc_args.x_ob).T @ Q @ (x[:2]-ioc_args.x_ob) \
                    + weight_kernel(x[:2],weights[1]) * (x[2:4]+u).T @ R @ (x[2:4]+u) \
                    + weight_kernel(x[:2],weights[2]) * u.T @ r @ u \
                    - weight_kernel(x[:2],weights[3]) * D * (u.T @ (x[:2]-ioc_args.x_ob))/(jnp.linalg.norm(u)*jnp.linalg.norm(x[:2]-ioc_args.x_ob)) \
                    + weight_kernel(x[:2],weights[4]) * E * jax.nn.sigmoid(-jnp.linalg.norm(x[:2]-ioc_args.x_ev)) )
        return cost

    # 終端コスト
    def term_cost(x,weights):
        cost = weight_kernel(x[:2],weights[5]) * 0.5 * (x[:2]-ioc_args.x_ob).T @ S @ (x[:2]-ioc_args.x_ob)
        return cost

    # ステージコストの微分（状態変数と入力による微分）
    grad_x_stage = jax.jit(jax.grad(stage_cost,0))
    grad_u_stage = jax.jit(jax.grad(stage_cost,1))
    hes_x_stage = jax.jit(jax.hessian(stage_cost,0))
    hes_u_stage = jax.jit(jax.hessian(stage_cost,1))
    hes_ux_stage = jax.jit(jax.jacfwd(jax.grad(stage_cost,1),0))

    # 終端コストの微分
    grad_x_term = jax.jit(jax.grad(term_cost))
    hes_x_term = jax.jit(jax.hessian(term_cost))

    def Backward(xs,us):
        Vxx = hes_x_term(xs[-1])
        Vx = grad_x_term(xs[-1])

        L = 0 # 尤度関数の値

        def Backward_body(carry, val):
            Vx, Vxx, L = carry
            x, u = val

            Ak = model_dfdx(x,u,ioc_args.dt)
            Bk = model_dfdu(x,u,ioc_args.dt)

            Qx = grad_x_stage(x,u,weights) + Vx @ Ak
            Qxx = hes_x_stage(x,u,weights) + Ak.T @ Vxx @ Ak

            Qu = grad_u_stage(x,u,weights) + Vx @ Bk
            Quu = hes_u_stage(x,u,weights) + Bk.T @ Vxx @ Bk

            Qux = hes_ux_stage(x,u,weights) + Bk.T @ Vxx @ Ak

            #Quuが正定かどうかの判定
            try:
                kekka = jnp.linalg.cholesky(Quu)
            except:
                #もし違ったら
                #正定化の為にまず固有値の最小値を特定する
                alpa = -jnp.amin(jnp.linalg.eig(Quu))
                Quu = Quu + (alpa + 1e-5) * jnp.eye(ioc_args.len_u) #正定化

            # 尤度関数の更新
            L = - (1/2*ioc_args.alpha) * Qu @ jnp.linalg.pinv(Quu) @ Qu.T \
                + (1/2) * jnp.log(jnp.linalg.det(Quu)) - (ioc_args.action_dim/2) * jnp.log(2*jnp.pi*ioc_args.alpha)

            K = - jnp.linalg.pinv(Quu) @ Qux # 閉ループゲインの計算
            d = - jnp.linalg.pinv(Quu) @ Qu.T # 開ループゲインの計算

            Vx = Qx + d.T @ Quu @ K + Qu @ K + d.T @ Qux # Vxの更新
            Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K # Vxxの更新

            return (Vx, Vxx, L), (K, d)
        
        
        carry, _ = jax.lax.scan(Backward_body, (Vx, Vxx, L), (jnp.flip(xs[:-1], 0), jnp.flip(us, 0)))
        L = carry[2]
        
        return L
    
    L = Backward(xs,us)
    
    return L


# データ格納バッファ
class ReplayBuffer():

    def __init__(self, total_timesteps, episode_length, obs_dim, action_dim):

        # バッファの定義
        num_episode = (total_timesteps // episode_length) + 1
        self.obs_buffer = np.empty((num_episode, episode_length + 1, obs_dim), dtype=np.float32)
        self.action_buffer = np.empty((num_episode, episode_length, action_dim), dtype=np.float32)
        #self.reward_buffer = np.empty((num_episode, episode_length, 1), dtype=np.float32)

        self.episode_length = episode_length
        self.iter = 0


    def add(self, obss, actions):

        # バッファにシーケンスを格納
        self.obs_buffer[self.iter] = obss
        self.action_buffer[self.iter] = actions
        #self.reward_buffer[self.iter] = rewards[:, None]
        self.iter += 1


    def sample(self, batch_size, horizon):

        # バッファから取得する要素の index をサンプリング
        idx = np.random.randint(self.iter, size=batch_size)
        idy = np.random.randint(self.episode_length - horizon, size=batch_size)

        obs_dim = self.obs_buffer.shape[2]
        action_dim = self.action_buffer.shape[2]

        # バッファからデータを取得
        obss = np.empty((horizon+1, batch_size, obs_dim), dtype=np.float32)
        actions = np.empty((horizon, batch_size, action_dim), dtype=np.float32)
        rewards = np.empty((horizon, batch_size, 1), dtype=np.float32)
        for t in range(horizon):
            obss[t] = self.obs_buffer[idx, idy+t]
            actions[t] = self.action_buffer[idx, idy+t]
            rewards[t] = self.reward_buffer[idx, idy+t]
        obss[horizon] = self.obs_buffer[idx, idy+horizon]
        return jnp.array(obss), jnp.array(actions), jnp.array(rewards)

rb = ReplayBuffer(total_timesteps, ioc_args.N, ioc_args.obs_dim, ioc_args.action_dim)