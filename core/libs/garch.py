import numpy as np
from dataclasses import dataclass
from typing import Callable
from scipy.stats import t , genpareto
from scipy.optimize import minimize
import pandas as pd

@dataclass
class TailParams:
    xi: float
    beta: float
    u: float        # seuil tel que TU le fournis (positif ou négatif, on gère)

@dataclass
class MixtureParams:
    df_student: float
    left: TailParams
    right: TailParams

def _normalize_thresholds(u_neg_input: float, u_pos_input: float):
    """
    Rend cohérents:
      - jonction gauche tL (négative)
      - jonction droite tR (positive)
      - magnitudes positives u_neg_mag, u_pos_mag pour les excédents GPD.
    Accepte u_neg_input / u_pos_input signés OU en magnitude.
    """
    # gauche
    if u_neg_input <= 0:
        tL = float(u_neg_input)
        u_neg_mag = float(abs(u_neg_input))
    else:
        # l’utilisateur a donné une magnitude; la jonction gauche est -u_neg
        tL = -float(u_neg_input)
        u_neg_mag = float(u_neg_input)

    # droite
    if u_pos_input >= 0:
        tR = float(u_pos_input)
        u_pos_mag = float(abs(u_pos_input))
    else:
        # l’utilisateur a donné une valeur négative; la jonction droite est +|u_pos|
        tR = float(abs(u_pos_input))
        u_pos_mag = float(abs(u_pos_input))

    if not (tL < 0 and tR > 0):
        raise ValueError("Seuils incohérents: on attend tL<0 et tR>0 après normalisation.")
    return tL, tR, u_neg_mag, u_pos_mag

def _gpd_pdf_y(y: np.ndarray, xi: float, beta: float) -> np.ndarray:
    """
    Densité GPD en excédent y >= 0.
    """
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    if xi == 0.0:
        support = (y >= 0)
        out[support] = (1.0 / beta) * np.exp(-y[support] / beta)
    else:
        support = (y >= 0) & (1.0 + xi * y / beta > 0)
        out[support] = (1.0 / beta) * (1.0 + xi * y[support] / beta) ** (-1.0/xi - 1.0)
    return out




def fit_gpd_tails_auto(residuals, seuils_test_droite=None, seuils_test_gauche=None, plot=True):
    """
    Ajuste deux lois de Pareto généralisée (GPD) avec détection automatique du seuil
    pour les queues droite et gauche, indépendamment.

    Paramètres
    ----------
    residuals : array-like
        Résidus standardisés ou innovations.

    seuils_test_droite : array-like or None
        Liste des seuils à tester pour la queue droite. Si None, générée automatiquement.

    seuils_test_gauche : array-like or None
        Liste des seuils à tester pour la queue gauche. Si None, générée automatiquement.

    plot : bool
        Si True, affiche les diagnostics graphiques.

    Retour
    ------
    dict : paramètres xi (forme), scale et seuil optimal pour chaque queue.
    """
    residuals = np.asarray(residuals)
    
    # === Queue droite ===
    right_data = residuals[residuals > 0]
    if seuils_test_droite is None:
        seuils_test_droite = np.linspace(np.percentile(right_data, 85), np.percentile(right_data, 99), 20)

    mrl_means_pos = []
    xi_list_pos = []
    scale_list_pos = []
    
    for u in seuils_test_droite:
        excess = right_data[right_data > u] - u
        if len(excess) < 30:  # Éviter des seuils trop extrêmes
            mrl_means_pos.append(np.nan)
            xi_list_pos.append(np.nan)
            scale_list_pos.append(np.nan)
            continue
        mrl_means_pos.append(np.mean(excess))
        xi, loc, scale = genpareto.fit(excess)
        xi_list_pos.append(xi)
        scale_list_pos.append(scale)
    
    # Choix du seuil optimal : début de la stabilité de xi
    xi_array = np.array(xi_list_pos)
    stable_index_pos = np.nanargmin(np.abs(np.gradient(xi_array)))
    u_pos_opt = seuils_test_droite[stable_index_pos]
    
    # Fit final pour la queue droite
    excess_pos_final = right_data[right_data > u_pos_opt] - u_pos_opt
    params_pos = genpareto.fit(excess_pos_final)

    # === Queue gauche ===
    left_data = -residuals[residuals < 0]
    if seuils_test_gauche is None:
        seuils_test_gauche = np.linspace(np.percentile(left_data, 85), np.percentile(left_data, 99), 20)

    mrl_means_neg = []
    xi_list_neg = []
    scale_list_neg = []
    
    for u in seuils_test_gauche:
        excess = left_data[left_data > u] - u
        if len(excess) < 30:
            mrl_means_neg.append(np.nan)
            xi_list_neg.append(np.nan)
            scale_list_neg.append(np.nan)
            continue
        mrl_means_neg.append(np.mean(excess))
        xi, loc, scale = genpareto.fit(excess)
        xi_list_neg.append(xi)
        scale_list_neg.append(scale)
    
    # Choix du seuil optimal : début de la stabilité de xi
    xi_array_neg = np.array(xi_list_neg)
    stable_index_neg = np.nanargmin(np.abs(np.gradient(xi_array_neg)))
    u_neg_opt = seuils_test_gauche[stable_index_neg]
    
    # Fit final pour la queue gauche
    excess_neg_final = left_data[left_data > u_neg_opt] - u_neg_opt
    params_neg = genpareto.fit(excess_neg_final)



    return {
        "right_tail": {
            "threshold": u_pos_opt,
            "xi": params_pos[0],
            "scale": params_pos[2]
        },
        "left_tail": {
            "threshold": -u_neg_opt,
            "xi": params_neg[0],
            "scale": params_neg[2]
        }
    }







def build_continuous_student_gpd_mixture(params: MixtureParams):
    """
    Construit f_mix(x) (pdf continue, normalisée) et sample_mixed(n),
    en gérant u_neg/u_pos fournis en signe OU en magnitude.
    """
    nu = float(params.df_student)

    # Normalise les seuils/jonctions et récupère les magnitudes d’excédents
    tL, tR, u_neg_mag, u_pos_mag = _normalize_thresholds(params.left.u, params.right.u)
    xi_L, beta_L = float(params.left.xi),  float(params.left.beta)
    xi_R, beta_R = float(params.right.xi), float(params.right.beta)

    # Student tronquée au centre
    s_pdf = lambda x: t.pdf(x, df=nu)
    P_center = max(1e-16, t.cdf(tR, df=nu) - t.cdf(tL, df=nu))
    s_tr  = lambda x: np.where((x >= tL) & (x <= tR), s_pdf(x) / P_center, 0.0)

    # GPD sur excédents positifs: y_R = x - tR ; y_L = tL - x
    g_r = lambda x: _gpd_pdf_y(x - tR, xi_R, beta_R) * (x > tR)
    g_l = lambda x: _gpd_pdf_y(tL - x, xi_L, beta_L) * (x < tL)

    # Continuité aux jonctions: alpha_C s_tr(tR) = alpha_R * 1/beta_R ; alpha_C s_tr(tL) = alpha_L * 1/beta_L
    s_tr_tR = float(s_tr(np.array([tR]))[0])
    s_tr_tL = float(s_tr(np.array([tL]))[0])
    denom = 1.0 + s_tr_tR * beta_R + s_tr_tL * beta_L
    if denom <= 0:
        raise ValueError("Paramètres incohérents (beta>0 ? seuils ?).")
    alpha_center = 1.0 / denom
    alpha_right  = alpha_center * s_tr_tR * beta_R
    alpha_left   = alpha_center * s_tr_tL * beta_L

    def f_mix(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return alpha_center * s_tr(x) + alpha_right * g_r(x) + alpha_left * g_l(x)

    def sample_mixed(n: int, rng: np.random.Generator = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        comps = rng.choice(3, size=n, p=[alpha_center, alpha_left, alpha_right])
        out = np.empty(n, dtype=float)
        # centre: accept-reject Student tronquée
        n_c = np.sum(comps == 0)
        if n_c > 0:
            S = []
            batch = max(4*n_c, 1000)
            while len(S) < n_c:
                z = t.rvs(df=nu, size=batch, random_state=rng)
                z = z[(z >= tL) & (z <= tR)]
                S.extend(z.tolist())
            out[comps == 0] = np.array(S[:n_c])
        # gauche: x = tL - y
        n_l = np.sum(comps == 1)
        if n_l > 0:
            y = genpareto.rvs(c=xi_L, scale=beta_L, size=n_l, random_state=rng)
            if xi_L < 0:
                y = np.minimum(y, -beta_L/xi_L - 1e-12)
            out[comps == 1] = tL - y
        # droite: x = tR + y
        n_r = np.sum(comps == 2)
        if n_r > 0:
            y = genpareto.rvs(c=xi_R, scale=beta_R, size=n_r, random_state=rng)
            if xi_R < 0:
                y = np.minimum(y, -beta_R/xi_R - 1e-12)
            out[comps == 2] = tR + y
        return out

    info = {
        "alpha_center": float(alpha_center), "alpha_left": float(alpha_left), "alpha_right": float(alpha_right),
        "tL": float(tL), "tR": float(tR), "u_neg_mag": float(u_neg_mag), "u_pos_mag": float(u_pos_mag)
    }
    return f_mix, sample_mixed, info







def generate_hybrid_innovations(conditional_variances, df_student, horizon, residuals, plot=False):
    """
    Génère ε_t avec une densité mixte continue:
      - Student-t(ν) tronquée sur [-u_neg, u_pos]
      - GPD à gauche (excédents y_L = t_L - x >= 0)
      - GPD à droite (excédents y_R = x - t_R >= 0)
    """
    # 1) Ajuste GPD automatiquement (ta fonction existante)
    gpd_params = fit_gpd_tails_auto(residuals, plot=False)
    
    # 2) Prépare les paramètres du mélange (NOTE: on passe u_neg/u_pos tels quels — signe accepté)
    mixture = MixtureParams(
        df_student = float(df_student),
        left  = TailParams(xi=float(gpd_params['left_tail']['xi']),
                           beta=float(gpd_params['left_tail']['scale']),
                           u=float(gpd_params['left_tail']['threshold'])),
        right = TailParams(xi=float(gpd_params['right_tail']['xi']),
                           beta=float(gpd_params['right_tail']['scale']),
                           u=float(gpd_params['right_tail']['threshold'])),
    )
    
    # 3) Construit la densité mixte continue + tirages
    f_mix, sample_mixed, mix_info = build_continuous_student_gpd_mixture(mixture)
    
    # 4) Tire z_t ~ mélange, puis ε_t = σ_t z_t
    z = sample_mixed(int(horizon))
    innovations = np.sqrt(conditional_variances[:horizon]) * z
    return innovations  # (ou return innovations, mix_info si tu veux logguer les poids/jonctions)







class NGARCHMixMLE:
    """
    NGARCH(1,1) avec vraisemblance sous densité mélange standardisée f_Z_std (Var=1):
        h_t = ω + α (ε_{t-1} - θ sqrt(h_{t-1}))^2 + β h_{t-1}
        ε_t = sqrt(h_t) Z_t,  Z_t ~ f_Z_std  (E[Z^2]=1)
    On fournit f_Z et sample_Z NON standardisés au constructeur; la classe standardise auto.
    """

    def __init__(self, f_Z, sample_Z, standardize=True, mc_moment_samples=200_000, h0=None):
        self.h0 = h0
        # Standardisation du mélange (pour avoir Var(Z)=1)
        if standardize:
            zs = sample_Z(mc_moment_samples)
            s2 = float(np.mean(zs**2))
            s  = float(np.sqrt(s2))
            if not np.isfinite(s) or s <= 0:
                raise ValueError("Standardisation NGARCH: moment 2 invalide.")
            self._scale_ = s
            self.f_Z_std = lambda z: s * f_Z(s*z)         # f_{Z*}(z) = s f_Z(s z)
            self.sample_Z_std = lambda n: sample_Z(n) / s # Z* = Z/s
        else:
            self._scale_ = 1.0
            self.f_Z_std = f_Z
            self.sample_Z_std = sample_Z

        self.params_ = None
        self.h_ = None
        self.z_ = None

    # --- Récurrence NGARCH
    def _recursion_h(self, eps, omega, alpha, beta, theta):
        T = len(eps)
        h = np.empty(T, dtype=float)
        h[0] = self.h0 if (self.h0 is not None) else float(np.var(eps))
        if h[0] <= 0 or not np.isfinite(h[0]):
            h[0] = float(np.var(eps) if np.var(eps) > 0 else 1.0)
        for t in range(1, T):
            rt = eps[t-1]
            ht_1 = h[t-1]
            h[t] = omega + alpha * (rt - theta*np.sqrt(ht_1))**2 + beta * ht_1
            if h[t] <= 0 or not np.isfinite(h[t]):
                return None
        return h

    def _negloglik(self, x, eps):
        omega, alpha, beta, theta = x
        # contraintes de base + stationnarité NGARCH: beta + alpha(1+theta^2) < 1
        if (omega <= 1e-12) or (alpha < 0) or (beta < 0) or (alpha + beta >= 0.9999):
            return 1e10
        if beta + alpha*(1.0 + theta**2) >= 0.9999:
            return 1e10
        h = self._recursion_h(eps, omega, alpha, beta, theta)
        if h is None:
            return 1e10
        z = eps / np.sqrt(h)
        pdf = self.f_Z_std(z)
        if np.any(pdf <= 0) or (not np.all(np.isfinite(pdf))):
            return 1e10
        nll = -np.sum(np.log(pdf) - 0.5*np.log(h))
        return nll if np.isfinite(nll) else 1e10

    def fit(self, residuals, x0=None):
        eps = np.asarray(residuals, float)
        if x0 is None:
            # init douce: theta=0, alpha=0.05, beta=0.9
            x0 = np.array([0.1*np.var(eps), 0.05, 0.9, 0.0], float)
        bounds = [(1e-12, None), (0.0, 1.0), (0.0, 0.9999), (None, None)]
        res = minimize(self._negloglik, x0, args=(eps,), method="L-BFGS-B", bounds=bounds)
        if not res.success:
            raise RuntimeError(f"MLE NGARCHMix échec : {res.message}")
        self.params_ = dict(omega=res.x[0], alpha=res.x[1], beta=res.x[2], theta=res.x[3])
        self.nll_ = float(res.fun)   # négative log-likelihood au maximum
        self._eps = eps              # conserve les résidus pour infos/AIC-BIC
        # Trajectoire finale
        h = self._recursion_h(eps, **self.params_)
        self.h_ = h
        self.z_ = eps / np.sqrt(h)
        return self

    def forecast(self, horizon, mode="mean", n_paths=2000, seed=None, eps_T=None, h_T=None):
        """
        Prévision h_{T+1:T+H}.
        - mean : espérance analytique (sans tirages)
        - mc   : simulation (tirages Z ~ f_Z_std)
        """
        if self.params_ is None or self.h_ is None:
            raise RuntimeError("Appelle d'abord fit().")
        ω, α, β, θ = self.params_['omega'], self.params_['alpha'], self.params_['beta'], self.params_['theta']

        if eps_T is None:
            eps_T = float(self.z_[-1] * np.sqrt(self.h_[-1]))
        if h_T is None:
            h_T = float(self.h_[-1])

        H = int(horizon)
        out = np.empty(H, float)

        if mode == "mean":
            # 1er pas (déterministe)
            h1 = ω + α*(eps_T - θ*np.sqrt(h_T))**2 + β*h_T
            out[0] = h1
            # À partir du 2e : E[h_{t+1}] = ω + (β + α(1+θ^2)) E[h_t]
            phi = β + α*(1.0 + θ**2)
            for k in range(1, H):
                h1 = ω + phi * h1
                out[k] = h1
            return out

        elif mode == "mc":
            rng = np.random.default_rng(seed)
            paths = np.empty((n_paths, H), float)
            for s in range(n_paths):
                curr = ω + α*(eps_T - θ*np.sqrt(h_T))**2 + β*h_T  # T+1
                paths[s, 0] = curr
                for k in range(1, H):
                    z = self.sample_Z_std(1)[0]
                    eps = np.sqrt(curr) * z
                    curr = ω + α*(eps - θ*np.sqrt(curr))**2 + β*curr
                    paths[s, k] = curr
            return paths.mean(axis=0), paths

        else:
            raise ValueError("mode ∈ {'mean','mc'}")


    def params_df(self):
        """
        Tableau des paramètres MLE et métriques dérivées pour NGARCH(1,1):
            h_t = ω + α (ε_{t-1} - θ sqrt(h_{t-1}))^2 + β h_{t-1}
        """
        if self.params_ is None:
            raise RuntimeError("Appelle fit() avant params_df().")
        ω  = float(self.params_['omega'])
        α  = float(self.params_['alpha'])
        β  = float(self.params_['beta'])
        θ  = float(self.params_['theta'])
    
        # Persistance NGARCH (variance finie si phi_ngarch < 1)
        phi_ngarch = β + α * (1.0 + θ**2)
        if phi_ngarch < 1.0:
            h_bar = ω / max(1e-12, (1.0 - phi_ngarch))
            half_life = np.log(0.5) / np.log(phi_ngarch) if 0.0 < phi_ngarch < 1.0 else np.inf
        else:
            h_bar = np.nan
            half_life = np.inf
    
        n = len(self.h_) if getattr(self, "h_", None) is not None else np.nan
        k = 4  # omega, alpha, beta, theta
        nll = getattr(self, "nll_", np.nan)
        aic = 2*k + 2*nll if np.isfinite(nll) and np.isfinite(n) else np.nan
        bic = k*np.log(n) + 2*nll if np.isfinite(nll) and np.isfinite(n) else np.nan
    
        data = {
            "omega": ω,
            "alpha": α,
            "beta":  β,
            "theta": θ,
            "persistence(beta+alpha*(1+theta^2))": phi_ngarch,
            "uncond_var(omega/(1-persistence))": h_bar,
            "half_life[periods]": half_life,
            "sample_size": n,
            "neg_loglik": nll,
            "AIC": aic,
            "BIC": bic,
        }
        return pd.DataFrame.from_dict(data, orient="index", columns=["value"])