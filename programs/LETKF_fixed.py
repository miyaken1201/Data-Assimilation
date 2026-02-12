# 固定インフレーションのLETKF実装
import numpy as np
from utils import rk4_step
class EnKF_LETKF:
    def __init__(self, N, F, dt, H=None, R=None, ensemble_size=20, obs_error_std=1.0, inflation_factor=1.02, loc_radius=5):
        """
        LETKFを初期化するクラス
        N: 状態変数の数
        F: 外部からの強制力
        dt: 時間ステップ
        H: 観測行列
        R: 観測誤差共分散行列
        ensemble_size: アンサンブルのサイズ
        obs_error_std: 観測誤差の標準偏差

        """
        self.N = N
        self.F = F
        self.dt = dt
        self.H = H if H is not None else np.eye(N,N)
        self.p = self.H.shape[0]
        self.R = R if R is not None else np.eye(self.p, self.p) * obs_error_std
        self.Z = None
        self.ensemble_mean = None
        self.ensemble_list = []
        self.P_analysis_list = []
        self.ensemble_size = ensemble_size
        self.obs_error_std = obs_error_std
        self.ensemble = None
        self.inflation_factor = inflation_factor
        self.L = self.localization_matrix(loc_radius=loc_radius)
        # Hが密行列(dense)の場合や特殊な配置の場合、obs_indicesの自動判定は注意が必要
        # ここでは「Hの各列がいずれかの観測に関与しているか」で判定している
        self.obs_indices = np.where(np.any(self.H, axis=0))[0]

    def initialize_ensemble(self, x0):
        np.random.seed(42)  # For reproducibility
        members = [x0 + np.random.normal(0, 1, self.N) for _ in range(self.ensemble_size)]
        self.ensemble = np.column_stack(members)
        np.random.seed() # シード値の固定を解除

    def localization_matrix(self, loc_radius):
        """Gaussian Functionでの局所化行列を作成する"""
        ### グリッド間距離の計算
        L = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                dist = min(abs(i - j), self.N - abs(i - j))
                if dist < 2*np.sqrt(10/3)*loc_radius:
                    # exp(- (dist / loc_radius)^2 / 2)  の形でガウシアン関数を適用
                    L[i, j] = np.exp(-0.5 * (dist / loc_radius) ** 2)
                else:
                    L[i, j] = 0.0
        return L
                    
    def forecast(self):
        """アンサンブルの予測ステップを実行する"""
        for i in range(self.ensemble_size):
            self.ensemble[:, i] = self.step_model(self.ensemble[:, i])
        self.ensemble_mean = np.mean(self.ensemble, axis=1, keepdims=True)
        self.Z = (self.ensemble - self.ensemble_mean) / np.sqrt(self.ensemble_size - 1)
    def step_model(self, x):
        """モデルの1ステップ予測を実行する"""
        return rk4_step(x, self.F, self.dt)

    def analysis(self, observation):
        """アンサンブルの解析ステップを実行する"""
        # (1) Preparation Step
        # Zb = self.Z * np.sqrt(self.inflation_factor)
        Zb = self.Z * self.inflation_factor
        Yb = self.H @ Zb
        diff = self.H @ observation.reshape(-1, 1) - self.H @ self.ensemble_mean
        # (2) Local Analysis Step
        L = self.L # 局所化行列の取得
        Xa = np.zeros_like(self.ensemble)
        # Rの逆行列を事前に計算
        R_inv = np.linalg.inv(self.R)
        
        # エラーチェック用のフラグ
        has_error = False
        
        for i in range(self.N):
            # (2.1) Eigenvalue Decomposition with Localization
            # 局所化行列の取得
            loc_weights = L[i, :]
            loc_weights_obs = loc_weights[self.obs_indices]          # 長さ = 観測次元
            R_loc_inv = R_inv * loc_weights_obs[:, None]
            Pa_tilde_inv = np.eye(self.ensemble_size) + Yb.T @ R_loc_inv @ Yb
            
            try:
                D, C = np.linalg.eigh(Pa_tilde_inv)
            except np.linalg.LinAlgError:
                if not has_error: # 最初のエラーだけ詳細を表示
                    print(f"LinAlgError at grid point {i}")
                    print(f"Pa_tilde_inv max: {np.max(Pa_tilde_inv)}, min: {np.min(Pa_tilde_inv)}")
                    print(f"Contains NaNs: {np.isnan(Pa_tilde_inv).any()}")
                    print(f"Contains Infs: {np.isinf(Pa_tilde_inv).any()}")
                    has_error = True
                # エラー時は更新せず予測値をそのまま使う（あるいは例外を投げる）
                # ここでは例外を再送出する
                raise

            D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
            Pa_tilde = C @ np.diag(1.0 / D) @ C.T
            pa_tilde_sqrt = C @ D_inv_sqrt @ C.T
            # (2.2) Update Ensemble
            T = Pa_tilde @ Yb.T @ R_loc_inv @ diff + np.sqrt(self.ensemble_size - 1)* pa_tilde_sqrt
            Xa[i, :] = self.ensemble_mean[i, 0] + Zb[i, :] @ T
        # (3) Collect analysis Ensemble of all model grid points
        self.ensemble = Xa
        self.ensemble_mean = np.mean(self.ensemble, axis=1, keepdims=True)
        Za = (self.ensemble - self.ensemble_mean) / np.sqrt(self.ensemble_size - 1)
        
        self.P_analysis_list.append((Za @ Za.T).copy().T)
        self.ensemble_list.append(self.ensemble.copy().T)

    def get_ensemble_list(self):
        return self.ensemble_list

    def get_P_analysis_list(self):
        return self.P_analysis_list  

    def calculate_rmse(self, true_states):
        ensemble_mean = np.mean(self.ensemble_list, axis=1)
        rmse = ensemble_mean - true_states
        return np.sqrt(np.mean(rmse**2, axis=1))

    def calculate_spread(self):
        spread_list = []
        for P_analysis in self.P_analysis_list:
            spread = np.sqrt(np.trace(P_analysis) / self.N)
            spread_list.append(spread)
        return spread_list
    def calculate_dfs_trace(self):
        """
        方法1: トレースを用いた計算
        DFS = trace( K H ) = trace( S (S + R)^-1 )
        """
        if self.Z is None: return 0.0
        
        # 1. 観測空間でのシグナル共分散 S の計算
        Zb = self.Z * self.inflation_factor
        Yb = self.H @ Zb 
        S = Yb @ Yb.T  # (H Pb H^T)
        
        # 2. DFS = trace( S @ (S+R)^-1 )
        denominator = S + self.R
        try:
            # (S+R)^-1 @ S を計算 (solveの方がinvより安定)
            KH = np.linalg.solve(denominator, S)
            dfs = np.trace(KH)
        except np.linalg.LinAlgError:
            dfs = 0.0
        return dfs

    def calculate_dfs_svd(self):
        """
        方法2: 特異値分解(SVD)を用いた計算
        DFS = sum( gamma_i / (1 + gamma_i) )
        gamma_i: 可観測行列 G = R^-1/2 H Pb^1/2 の特異値の二乗
        """
        if self.Z is None: return 0.0
        
        Zb = self.Z * self.inflation_factor # アンサンブル摂動 (N, m)
        
        # 1. Rの平方根の逆行列 R^-1/2 を計算
        # (Rが対角行列なら単に 1/sqrt(diag) ですが、汎用的に固有値分解で計算します)
        evals_R, evecs_R = np.linalg.eigh(self.R)
        # R^-1/2 = E * D^-1/2 * E^T
        R_inv_sqrt = evecs_R @ np.diag(1.0 / np.sqrt(evals_R)) @ evecs_R.T
        
        # 2. 白色化された摂動行列 G_ens の作成
        # G_ens = R^-1/2 @ H @ Zb
        G_ens = R_inv_sqrt @ (self.H @ Zb) # サイズ: (p, m)
        
        # 3. 特異値分解 (SVD)
        # G_ens の特異値 s を取得 (これが G の特異値に相当)
        s = np.linalg.svd(G_ens, compute_uv=False)
        
        # 4. 固有値 gamma = s^2 を用いてDFSを算出
        gamma = s**2
        dfs = np.sum(gamma / (1 + gamma))
        
        return dfs