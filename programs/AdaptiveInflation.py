def run_adaptive_filter(prior_var, N, F, dt, dense_H, ens_member_size, obs_error_std, loc_radius,
                        observations, true_states, initial_state, spinup_cycles, steps_per_assim):
    import numpy as np
    from tqdm import tqdm
    from LETKF_adaptive import EnKF_LETKF_Adaptive

    filt = EnKF_LETKF_Adaptive(
        N, F, dt, dense_H,
        ensemble_size=ens_member_size,
        obs_error_std=obs_error_std,
        loc_radius=loc_radius,
        prior_inflation_var=prior_var
    )
    filt.initialize_ensemble(initial_state)

    for cycle in tqdm(range(observations.shape[0]), desc=f"Assimilation cycles (adaptive {np.sqrt(prior_var):.3f}^2)"):
        for _ in range(steps_per_assim):
            filt.forecast()
        filt.analysis(observations[cycle])

    inflation_params = np.array(filt.get_inflation_parameter_list())
    inflation_params_mean = np.mean(inflation_params, axis=1)
    rmse_series = filt.calculate_rmse(true_states)
    spread_series = filt.calculate_spread()
    ens_list = np.array(filt.get_ensemble_list())
    P_list = np.array(filt.get_P_analysis_list())

    mid = N // 2
    r_g = np.mean(rmse_series[spinup_cycles:])
    s_g = np.mean(spread_series[spinup_cycles:])
    mean_ens_land = np.mean(ens_list[spinup_cycles:, :, :mid], axis=1)
    true_land = true_states[spinup_cycles:, :mid]
    r_l = np.mean(np.sqrt(np.mean((mean_ens_land - true_land) ** 2, axis=1)))
    s_l = np.mean([np.sqrt(np.trace(P[:mid, :mid]) / mid) for P in P_list[spinup_cycles:]])
    mean_ens_ocean = np.mean(ens_list[spinup_cycles:, :, mid:], axis=1)
    true_ocean = true_states[spinup_cycles:, mid:]
    r_o = np.mean(np.sqrt(np.mean((mean_ens_ocean - true_ocean) ** 2, axis=1)))
    s_o = np.mean([np.sqrt(np.trace(P[mid:, mid:]) / mid) for P in P_list[spinup_cycles:]])

    inflation_params_land = (np.mean(inflation_params[spinup_cycles:, :mid]) - 1) * 100
    inflation_params_ocean = (np.mean(inflation_params[spinup_cycles:, mid:]) - 1) * 100

    return {
        "prior_var": prior_var,
        "inflation_params_mean": inflation_params_mean,
        "inflation_params": inflation_params,  # 追加: 後段の保存/再利用用
        "rmse": rmse_series,
        "spread": spread_series,
        "metrics": [r_g, s_g, r_l, s_l, r_o, s_o],
        "inflation_params_land": inflation_params_land,
        "inflation_params_ocean": inflation_params_ocean,
    }

def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pathlib import Path
    from utils import StateArray, make_dense_H
    from LETKF_fixed import EnKF_LETKF
    from LETKF_adaptive import EnKF_LETKF_Adaptive
    # パラメータ設定
    N = 40          # 状態変数の次元
    F = 8.0         # 外部からの強制力
    dt_model = 0.005       # モデル時間ステップ
    dt_assim = 0.05        # 同化間隔
    dt_per_day = 0.2
    total_year = 110      # 総実験年数
    spinup_year = 10    # スピンアップ期間
    steps_per_assim = int(round(dt_assim / dt_model))
    total_cycles = int(total_year * 365 * (dt_per_day / dt_assim))
    spinup_cycles = int(spinup_year * 365 * (dt_per_day / dt_assim))
    obs_error_std = 1.0
    ens_member_size = 10  # アンサンブルメンバー数
    loc_radius = 3     # 局所化半径
    prior_inflation_var_list = [0.01**2, 0.02**2, 0.04**2, 0.08**2]
    dense_H = make_dense_H(N, N//2, dense=False)

    cache_dir = Path("cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    true_obs_cache = cache_dir / "true_obs.npz"

    # 真値の生成
    if true_obs_cache.exists():
        data = np.load(true_obs_cache)
        true_states = data["true_states"]
        observations = data["observations"]
        print("Loaded cached true_states/observations.")
    else:
        true_state = StateArray(N, F, dt_model, method='rk4', random_seed=1234)
        true_states = []
        for cycle in tqdm(range(100 * 365 * int(dt_per_day / dt_assim)), desc="Spin-up true states"):
            for _ in range(steps_per_assim):
                true_state.step()
        for cycle in tqdm(range(total_cycles), desc="Generate true states"):
            for _ in range(steps_per_assim):
                true_state.step()
            true_states.append(true_state.x.copy())
        true_states = np.array(true_states)  # shape (total_cycles, N)

        # 真値に観測ノイズを加えた観測データの生成
        np.random.seed(1234)        #seed値の固定
        observations = true_states + np.random.normal(0, obs_error_std, true_states.shape)
        np.savez_compressed(true_obs_cache, true_states=true_states, observations=observations)
        print("Data generation completed and cached.")

    # 通常のLETKFの初期化(比較用、1.5%固定インフレーション)
    enkf_letkf = EnKF_LETKF(N, F, dt_model, dense_H, ensemble_size=ens_member_size, obs_error_std=obs_error_std, inflation_factor=1.015, loc_radius=loc_radius)

    # LETKF with Adaptive Covariance Inflationの初期化 (prior varianceごとに実行)
    adaptive_filters = {}
    for prior_var in prior_inflation_var_list:
        adaptive_filters[prior_var] = EnKF_LETKF_Adaptive(N, F, dt_model, dense_H, ensemble_size=ens_member_size, obs_error_std=obs_error_std, loc_radius=loc_radius, prior_inflation_var=prior_var)

    # アンサンブルの初期化
    initial_state = true_states[0] + np.random.normal(0, 1.0, N)
    enkf_letkf.initialize_ensemble(initial_state)
    for filt in adaptive_filters.values():
        filt.initialize_ensemble(initial_state)

    # データ同化ループ (Fixedのみ)
    fixed_cache = cache_dir / "fixed_assimilation.npz"
    if fixed_cache.exists():
        data = np.load(fixed_cache)
        ensemble_list = data["ensemble_list"]
        P_analysis_list = data["P_analysis_list"]
        rmse = data["rmse"]
        spread = data["spread"]
        print("Loaded cached fixed assimilation.")
    else:
        for cycle in tqdm(range(total_cycles), desc="Assimilation cycles (fixed)"):
            for _ in range(steps_per_assim):
                enkf_letkf.forecast()
            enkf_letkf.analysis(observations[cycle])
        print("Data assimilation completed.")
        ensemble_list = np.array(enkf_letkf.get_ensemble_list())
        P_analysis_list = np.array(enkf_letkf.get_P_analysis_list())
        rmse = enkf_letkf.calculate_rmse(true_states)
        spread = enkf_letkf.calculate_spread()
        np.savez_compressed(
            fixed_cache,
            ensemble_list=ensemble_list,
            P_analysis_list=P_analysis_list,
            rmse=rmse,
            spread=spread
        )

    # Adaptive同化はprior_inflation_varごとに並列実行
    adaptive_results = {}
    all_adaptive_metrics = {}
    pending_priors = []
    for prior_var in prior_inflation_var_list:
        cache_path = cache_dir / f"adaptive_{prior_var:.6f}.npz"
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            adaptive_results[prior_var] = {
                "inflation_params_mean": data["inflation_params_mean"],
                "inflation_params": data["inflation_params"],
                "rmse": data["rmse"],
                "spread": data["spread"],
            }
            all_adaptive_metrics[prior_var] = data["metrics"]
            label = f"{np.sqrt(prior_var):.3f}^2"
            print(f"{label} Land Inflation Mean (post spin-up): {data['inflation_params_land']:.4f}%")
            print(f"{label} Ocean Inflation Mean (post spin-up): {data['inflation_params_ocean']:.4f}%")
        else:
            pending_priors.append(prior_var)

    if pending_priors:
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    run_adaptive_filter,
                    prior_var, N, F, dt_model, dense_H, ens_member_size, obs_error_std, loc_radius,
                    observations, true_states, initial_state, spinup_cycles, steps_per_assim
                ): prior_var
                for prior_var in pending_priors
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Adaptive filters"):
                result = fut.result()
                prior_var = result["prior_var"]
                adaptive_results[prior_var] = {
                    "inflation_params_mean": result["inflation_params_mean"],
                    "inflation_params": result["inflation_params"],
                    "rmse": result["rmse"],
                    "spread": result["spread"],
                }
                all_adaptive_metrics[prior_var] = result["metrics"]
                cache_path = cache_dir / f"adaptive_{prior_var:.6f}.npz"
                np.savez_compressed(
                    cache_path,
                    inflation_params_mean=result["inflation_params_mean"],
                    inflation_params=result["inflation_params"],
                    rmse=result["rmse"],
                    spread=result["spread"],
                    metrics=np.array(result["metrics"]),
                    inflation_params_land=result["inflation_params_land"],
                    inflation_params_ocean=result["inflation_params_ocean"],
                )
                label = f"{np.sqrt(prior_var):.3f}^2"
                print(f"{label} Land Inflation Mean (post spin-up): {result['inflation_params_land']:.4f}%")
                print(f"{label} Ocean Inflation Mean (post spin-up): {result['inflation_params_ocean']:.4f}%")
    # adptive_results と all_adaptive_metricsをprior_varが昇順にソート
    adaptive_results = dict(sorted(adaptive_results.items()))
    all_adaptive_metrics = {k: all_adaptive_metrics[k] for k in sorted(all_adaptive_metrics.keys())}

    # 結果の評価
    years = np.arange(total_cycles) * (dt_assim / dt_per_day) / 365
    tick_step = 10
    ## 全期間のInflation値の全地点平均をプロット
    color = ['black', 'red', 'green', 'deepskyblue']
    plt.figure(figsize=(8, 6))
    for prior_var, metrics in adaptive_results.items():
        label = f"{np.sqrt(prior_var):.3f}^2"
        plt.plot(years, metrics["inflation_params_mean"], label=label, color=color.pop(0), lw=0.25, alpha=0.8)
    plt.xlim(0, total_year)
    plt.xticks(np.arange(0, total_year + 1, tick_step))
    plt.ylim(1.0, 1.1)
    plt.yticks(np.arange(1.0, 1.1, 0.01))
    plt.grid()
    plt.xlabel('year')
    plt.ylabel('global mean inflation parameter')
    plt.title('Entire experiment global mean inflation parameter')
    plt.legend()
    plt.savefig('Global_Mean_Inflation_Entire.png', dpi=600)

    ## 直近10年間 (もしくは利用可能な最終期間) のInflation値平均を拡大表示
    color = ['black', 'red', 'green', 'deepskyblue']
    plt.figure(figsize=(8, 6))
    window_years = min(10, total_year)
    window_start = max(total_year - window_years, 0)
    window_tick_step = 1
    window_ticks = np.arange(window_start, total_year + 1, window_tick_step)
    if window_ticks.size == 0:
        window_ticks = np.array([window_start, total_year])
    for prior_var, metrics in adaptive_results.items():
        label = f"{np.sqrt(prior_var):.3f}^2"
        plt.plot(years, metrics["inflation_params_mean"], label=label, color=color.pop(0), lw=0.5, alpha=0.8)
    plt.xlim(window_start, total_year)
    plt.xticks(window_ticks)
    plt.ylim(1.0, 1.1)
    plt.yticks(np.arange(1.0, 1.1, 0.01))
    plt.grid()
    plt.xlabel('year')
    plt.ylabel('global mean inflation parameter')
    plt.title('Global mean inflation parameter in the last decade')
    plt.legend()
    plt.savefig('Global_Mean_Inflation_Last_Decade.png', dpi=600)

    ## スピンアップ期間を除いたInflation平均 (Land/Ocean) をprior varianceごとに表示
    # for prior_var, metrics in adaptive_results.items():
    #     inflation_params = metrics["inflation_params"]
    #     inflation_params_land = (np.mean(inflation_params[spinup_cycles:, 0:20]) - 1) * 100
    #     inflation_params_ocean = (np.mean(inflation_params[spinup_cycles:, 20:40]) - 1) * 100
    #     label = f"{np.sqrt(prior_var):.3f}^2"
    #     print(f"{label} Land Inflation Mean (post spin-up): {inflation_params_land:.4f}%")
    #     print(f"{label} Ocean Inflation Mean (post spin-up): {inflation_params_ocean:.4f}%")


    ## --------------------------------------------------------------------------------
    ## 比較データの準備 (Fixed LETKF)
    ## --------------------------------------------------------------------------------
    # スピンアップ期間を除いた100年分のRMSEとSpreadをGlobal/Land/Oceanで評価
    # （ensemble_list/P_analysis_list/rmse/spread はキャッシュ済み）
    if not fixed_cache.exists():
        rmse = enkf_letkf.calculate_rmse(true_states)
        spread = enkf_letkf.calculate_spread()

    # Fixed results
    rmse_global_fixed = np.mean(rmse[spinup_cycles:])
    spread_global_fixed = np.mean(spread[spinup_cycles:])
    rmse_land_fixed = np.mean(np.sqrt(np.mean((np.mean(ensemble_list[spinup_cycles:, :, 0:20], axis=1) - true_states[spinup_cycles:, 0:20])**2, axis=1)))
    rmse_ocean_fixed = np.mean(np.sqrt(np.mean((np.mean(ensemble_list[spinup_cycles:, :, 20:40], axis=1) - true_states[spinup_cycles:, 20:40])**2, axis=1)))
    spread_land_fixed = np.mean([np.sqrt(np.trace(P_analysis[0:20, 0:20]) / 20) for P_analysis in P_analysis_list[spinup_cycles:]])
    spread_ocean_fixed = np.mean([np.sqrt(np.trace(P_analysis[20:40, 20:40]) / 20) for P_analysis in P_analysis_list[spinup_cycles:]])

    fixed_results = [rmse_global_fixed, spread_global_fixed, rmse_land_fixed, spread_land_fixed, rmse_ocean_fixed, spread_ocean_fixed]

    ## --------------------------------------------------------------------------------
    ## 比較データの準備 (Adaptive LETKF 全パターン)
    ## --------------------------------------------------------------------------------
    # all_adaptive_metrics はキャッシュ/実行結果から既に作成済み
    # all_adaptive_metrics = {}  # ← 上書きしない
    #
    # for prior_var, filt in adaptive_filters.items():
    #     ens_list = np.array(filt.get_ensemble_list())  # 未実行のため空/1次元になる
    #     P_list = np.array(filt.get_P_analysis_list())
    #     ...
    #     all_adaptive_metrics[prior_var] = [r_g, s_g, r_l, s_l, r_o, s_o]


    ## --------------------------------------------------------------------------------
    ## グラフ描画1: RMSEとSpreadの絶対値比較
    ## --------------------------------------------------------------------------------
    labels = ['RMSE\nGlobal', 'Spread\nGlobal', 'RMSE\nLand', 'Spread\nLand', 'RMSE\nOcean', 'Spread\nOcean']
    x = np.arange(len(labels))
    color = ['black', 'red', 'green', 'deepskyblue']
    # バーの設定：1.5%固定 + Prior種類の数
    num_bars = 1 + len(prior_inflation_var_list)
    total_width = 0.8  # グループ全体の幅
    single_width = total_width / num_bars

    plt.figure(figsize=(8, 6))

    # Fixed Baseline
    # オフセット位置の計算: 中心から見て左端へ
    start_offset = -total_width / 2 + single_width / 2
    plt.bar(x + start_offset, fixed_results, single_width, label='1.5% fixed', alpha=0.9, color='gray')

    # Adaptive Results (Loop)
    sorted_priors = sorted(prior_inflation_var_list)
    for i, pv in enumerate(sorted_priors):
        offset = start_offset + (i + 1) * single_width
        label_str = f"{np.sqrt(pv):.3f}^2"
        plt.bar(x + offset, all_adaptive_metrics[pv], single_width, label=label_str, alpha=0.9, color=color.pop(0))

    plt.xticks(x, labels)
    plt.ylabel('Value')
    plt.title('RMSE and Spread Comparison (post spin-up)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 3) # 必要に応じて調整
    plt.yticks(np.arange(0, 3.1, 0.5))
    plt.tight_layout()
    plt.savefig('RMSE_Spread_Comparison.png', dpi=600)

    ## --------------------------------------------------------------------------------
    ## グラフ描画2: 1.5%固定インフレーションとのRMSE改善率 (Global/Land/Ocean)
    ## --------------------------------------------------------------------------------
    # 指標のラベル (RMSEのみ)
    imp_labels = ['RMSE(Global)', 'RMSE(Land)', 'RMSE(Ocean)']
    x_imp = np.arange(len(imp_labels))
    color = ['black', 'red', 'green', 'deepskyblue']

    plt.figure(figsize=(8, 6))

    # RMSEのインデックスは 0, 2, 4
    rmse_indices = [0, 2, 4]

    # FixedのRMSE値
    rmse_fixed_vals = [fixed_results[idx] for idx in rmse_indices]

    # バーの設定：Prior種類の数だけ（Fixedは基準なのでプロットしない、あるいは0として扱う）
    num_bars_imp = len(prior_inflation_var_list)
    total_width_imp = 0.8
    single_width_imp = total_width_imp / num_bars_imp

    # オフセット位置の初期化
    start_offset_imp = -total_width_imp / 2 + single_width_imp / 2

    for i, pv in enumerate(sorted_priors):
        # 各Adaptive設定のRMSEを取得
        metrics = all_adaptive_metrics[pv]
        rmse_adaptive_vals = [metrics[idx] for idx in rmse_indices]
        
        # 改善率の計算: (Fixed - Adaptive) / Fixed * 100
        improvements = []
        for f_val, a_val in zip(rmse_fixed_vals, rmse_adaptive_vals):
            imp_pct = (f_val - a_val) / f_val * 100
            improvements.append(imp_pct)
        
        offset = start_offset_imp + i * single_width_imp
        label_str = f"{np.sqrt(pv):.3f}^2"
        plt.bar(x_imp + offset, improvements, single_width_imp, label=label_str, alpha=0.9, color=color.pop(0))

    plt.xticks(x_imp, imp_labels)
    plt.ylabel('Improvement (%)')
    plt.title('RMSE Improvement by Adaptive Inflation over Fixed 1.5% Inflation')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(-2, 10) # 必要に応じて調整
    plt.yticks(np.arange(-2, 11, 2))
    plt.tight_layout()
    plt.savefig('RMSE_Improvement_Comparison.png', dpi=600)

if __name__ == "__main__":
    main()