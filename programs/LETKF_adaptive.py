# Miyoshi (2011) の適応的インフレーション対応 LETKF の実装
import numpy as np
from utils import rk4_step
class EnKF_LETKF_Adaptive:
    def __init__(self, N, F, dt, H=None, R=None, ensemble_size=20, 
                 obs_error_std=1.0, 
                 initial_inflation=1.0,  # 初期インフレーション値
                 prior_inflation_var=0.04**2, # Miyoshi2011推奨の事前分散 (v^b)
                 loc_radius=3):
        """
        Miyoshi (2011) の適応的インフレーション対応 LETKF
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
        self.inflation_parameter_list = []
        self.ensemble_size = ensemble_size
        self.obs_error_std = obs_error_std
        self.ensemble = None
        
        # --- Adaptive Inflation 用の変数を初期化 ---
        # インフレーション値はグリッドごとに異なる値を持つ (ベクトル)
        self.inflation_parameter = np.ones(self.N) * initial_inflation
        # インフレーションの事前分散 (固定パラメータとして扱うのが一般的)
        self.prior_inflation_var = prior_inflation_var 
        
        self.L = self.localization_matrix(loc_radius=loc_radius)
        self.obs_indices = np.where(np.any(self.H, axis=0))[0]

    def initialize_ensemble(self, x0):
        np.random.seed(42)
        members = [x0 + np.random.normal(0, 1, self.N) for _ in range(self.ensemble_size)]
        self.ensemble = np.column_stack(members)
        np.random.seed()

    def localization_matrix(self, loc_radius):
        """Gaussian Functionでの局所化行列を作成する"""
        L = np.zeros((self.N, self.N))
        # 注: 実装効率化のため、ここでは単純な距離計算にしています
        for i in range(self.N):
            for j in range(self.N):
                dist = min(abs(i - j), self.N - abs(i - j))
                if dist < 2*np.sqrt(10/3)*loc_radius:
                    L[i, j] = np.exp(-0.5 * (dist / loc_radius) ** 2)
                else:
                    L[i, j] = 0.0
        return L
                    
    def forecast(self):
        """アンサンブルの予測ステップ"""
        for i in range(self.ensemble_size):
            self.ensemble[:, i] = self.step_model(self.ensemble[:, i])
        self.ensemble_mean = np.mean(self.ensemble, axis=1, keepdims=True)
        # 摂動の計算 (正規化込み)
        self.Z = (self.ensemble - self.ensemble_mean) / np.sqrt(self.ensemble_size - 1)

    def step_model(self, x):
        """モデルの1ステップ予測 (RK4などを想定)"""
        # ユーザー提供コードに rk4_step が含まれていなかったため、ダミー実装または外部関数と想定
        # ここでは外部関数 rk4_step がある前提で記述します
        return rk4_step(x, self.F, self.dt)

    def analysis(self, observation):
        """
        適応的インフレーションを含む解析ステップ
        Miyoshi (2011) の Eq.6, Eq.14, Eq.15 を実装
        """
        # (1) Preparation Step
        # 空間的に異なるインフレーション値を適用 (Broadcasting)
        # self.Z: (N, m), self.inflation_parameter: (N,) -> (N, 1)
        Zb = self.Z * self.inflation_parameter[:, np.newaxis]
        
        Yb = self.H @ Zb
        # Innovation vector d = y - Hx_b
        diff = self.H @ observation.reshape(-1, 1) - self.H @ self.ensemble_mean
        
        # (2) Local Analysis Step
        L = self.L
        Xa = np.zeros_like(self.ensemble)
        R_inv = np.linalg.inv(self.R)
        
        # 次のステップのために更新されたインフレーション値を格納する配列
        new_inflation_parameter = self.inflation_parameter.copy()
        self.inflation_parameter_list.append(new_inflation_parameter)

        has_error = False
        
        for i in range(self.N):
            # --- 局所化重みの取得 ---
            loc_weights = L[i, :]
            loc_weights_obs = loc_weights[self.obs_indices]
            
            # 局所化された Rの逆行列: R_loc_inv = R^-1 * rho
            # 対角行列を仮定した高速化 (一般的な実装)
            R_loc_inv = R_inv * loc_weights_obs[:, None] 

            # --- Adaptive Inflation Estimation (Miyoshi 2011) ---
            # 有効観測点数 p_eff = tr(rho)
            p_eff = np.sum(loc_weights_obs)

            if p_eff > 1e-4: # 観測の影響がある場合のみ計算
                # Eq.14: 分子の計算 (Innovation statistics)
                # tr(d d^T rho R^-1) = d^T (rho R^-1) d
                # diff は (p, 1) なので結果はスカラー
                numerator_term = (diff.T @ R_loc_inv @ diff)[0, 0]
                alpha_o_numerator = numerator_term - p_eff

                # Eq.14: 分母の計算 (Forecast spread statistics)
                # tr(H P H^T rho R^-1) = tr(Y_b Y_b^T R_loc_inv) = tr(Y_b^T R_loc_inv Y_b)
                # Yb: (p, m), R_loc_inv: (p, p)
                # P = Z Z^T なので、ここでは Yb Yb^T が HPH^T に相当
                denom_matrix = Yb.T @ R_loc_inv @ Yb
                denominator = np.trace(denom_matrix)

                if denominator > 1e-10:
                    # 観測から推定されたインフレーション値 alpha_o (Eq.14)
                    alpha_o = alpha_o_numerator / denominator
                    
                    # # 異常値の回避 (例えば負の値や極端な値を制限)
                    # # 論文では言及が少ないが、実装上は下限1.0などを設けることが多い
                    # alpha_o = max(1.0, alpha_o)

                    # 推定分散 v_o (Eq.15)
                    # 論文の改良: alpha_true の代わりに alpha_b (prior) を使用
                    alpha_b = self.inflation_parameter[i]
                    
                    # v_o = (2 / p_eff) * ((alpha * denom + p_eff) / denom)^2
                    term_inside = (alpha_b * denominator + p_eff) / denominator
                    v_o = (2.0 / p_eff) * (term_inside ** 2)

                    # Kalman Filter Update for Inflation (Eq.6)
                    v_b = self.prior_inflation_var
                    
                    # K = v_b / (v_b + v_o)
                    weight = v_b / (v_b + v_o)
                    
                    # alpha_a = (1-K)alpha_b + K*alpha_o
                    # 数式変形: alpha_a = alpha_b + K * (alpha_o - alpha_b)
                    new_alpha = alpha_b + weight * (alpha_o - alpha_b)
                    
                    # # 極端なインフレーションを防ぐリミッター (任意だが推奨)
                    # new_alpha = np.clip(new_alpha, 1.0, 1.5) # 例: 最大50%増まで
                    
                    new_inflation_parameter[i] = new_alpha

            # --- Standard LETKF Analysis ---
            # Pa_tilde_inv = (m-1)I + Yb^T R_loc_inv Yb
            # 注: self.Z は sqrt(m-1) で割られているため、
            # 通常のLETKFの式 (m-1)I / rho + ... ではなく、
            # I + Yb^T R_loc_inv Yb となる (Yb自体が正規化されているため)
            Pa_tilde_inv = np.eye(self.ensemble_size) + Yb.T @ R_loc_inv @ Yb
            
            try:
                D, C = np.linalg.eigh(Pa_tilde_inv)
                D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
                D_inv = np.diag(1.0 / D)
                
                # Pa_tilde = C D^-1 C^T
                Pa_tilde = C @ D_inv @ C.T
                # pa_tilde_sqrt = C D^-0.5 C^T
                pa_tilde_sqrt = C @ D_inv_sqrt @ C.T
                
                # Update weights T
                # T = Pa_tilde Yb^T R_loc_inv d + sqrt(m-1) * Pa_tilde_sqrt
                w_mean = Pa_tilde @ Yb.T @ R_loc_inv @ diff
                W_pert = np.sqrt(self.ensemble_size - 1) * pa_tilde_sqrt
                
                T = w_mean + W_pert
                
                # Update Analysis Ensemble
                # Xa = x_bar + Zb * T
                Xa[i, :] = self.ensemble_mean[i, 0] + Zb[i, :] @ T

            except np.linalg.LinAlgError:
                if not has_error:
                    print(f"LinAlgError at grid point {i}")
                    has_error = True
                # エラー時は更新せず、予測値（事前分布）をそのまま採用
                Xa[i, :] = self.ensemble[i, :]

        # (3) Update State and Inflation
        self.ensemble = Xa
        self.inflation_parameter = new_inflation_parameter # 次のステップのためにインフレーション値を更新
        
        self.ensemble_mean = np.mean(self.ensemble, axis=1, keepdims=True)
        # 解析誤差共分散の保存などは必要に応じて記述
        Za = (self.ensemble - self.ensemble_mean) / np.sqrt(self.ensemble_size - 1)
        self.P_analysis_list.append((Za @ Za.T).copy().T)
        self.ensemble_list.append(self.ensemble.copy().T)
    
    def get_P_analysis_list(self):
        return self.P_analysis_list
    
    def get_ensemble_list(self):
        return self.ensemble_list
    
    def get_inflation_parameter_list(self):
        return self.inflation_parameter_list

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