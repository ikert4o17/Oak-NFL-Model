"""Historical V11 weather/acclimation diagnostic."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from oak_nfl.data.nflverse import load_schedules
from oak_nfl.weather import build_temperature_acclimation, build_weather_features
TRAIN_END=2022; HOLDOUT_START=2023; HOLDOUT_END=2025

def _prepare_games():
    games=load_schedules(list(range(2000,HOLDOUT_END+1))).copy(); games=games[games["game_type"].isin(["REG","WC","DIV","CON","SB"])].copy()
    for c in ["home_score","away_score","total_line"]: games[c]=pd.to_numeric(games[c],errors="coerce")
    games["temperature_f"]=pd.to_numeric(games["temp"],errors="coerce"); games["wind_mph"]=pd.to_numeric(games["wind"],errors="coerce"); games["precipitation_in"]=0.0
    games["is_dome"]=games["roof"].fillna("").astype(str).str.lower().isin(["dome","closed"]); games["actual_total"]=games.home_score+games.away_score
    w=build_weather_features(games[["game_id","temperature_f","wind_mph","precipitation_in","is_dome"]])
    a=build_temperature_acclimation(games[["game_id","season","week","home_team","away_team","temperature_f","is_dome"]])
    return games.merge(w,on="game_id",suffixes=("","_weather")).merge(a,on="game_id",how="left")

def _metrics(a,p): return {"games":float(len(a)),"mae":float(mean_absolute_error(a,p)),"rmse":float(np.sqrt(mean_squared_error(a,p)))}
def _fit(train,holdout,features):
    target=train.actual_total-train.total_line
    prep=ColumnTransformer([("numeric",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),features)],remainder="drop")
    model=Pipeline([("prep",prep),("ridge",Ridge(alpha=10.0))]); model.fit(train,target)
    return holdout.total_line.to_numpy()+model.predict(holdout)

def _threshold_report(games):
    outdoor=games[games.outdoor.eq(1)].copy(); outdoor["market_residual"]=outdoor.actual_total-outdoor.total_line
    masks={"wind_15_plus":outdoor.wind_mph.ge(15),"wind_20_plus":outdoor.wind_mph.ge(20),"below_32":outdoor.temperature_f.lt(32),"below_32_away_recent_55_plus":outdoor.temperature_f.lt(32)&outdoor.away_recent_temp.ge(55),"below_35_away_recent_60_plus":outdoor.away_warm_team_freeze.eq(1),"above_85":outdoor.temperature_f.gt(85),"above_85_away_recent_60_minus":outdoor.away_cold_team_heat.eq(1)}
    print("\n=== EXTREME CONDITION DIAGNOSTIC ===")
    for name,mask in masks.items():
        x=outdoor[mask].dropna(subset=["market_residual"]); print(name,{"games":len(x),"avg_total":round(x.actual_total.mean(),3),"market_residual":round(x.market_residual.mean(),3)})

def run():
    games=_prepare_games().dropna(subset=["actual_total","total_line"]); train=games[games.season.le(TRAIN_END)]; holdout=games[games.season.between(HOLDOUT_START,HOLDOUT_END)]
    print("=== OAK V11 WEATHER / ACCLIMATION DIAGNOSTIC ==="); print(f"train games: {len(train)} | holdout games: {len(holdout)}"); print("MARKET TOTAL CONTROL",_metrics(holdout.actual_total,holdout.total_line))
    sets={
      "wind":["wind_over_10","wind_over_15"],
      "absolute_temperature":["cold_below_32","heat_above_85"],
      "home_climate_acclimation":["home_cold_shock","away_cold_shock","home_heat_shock","away_heat_shock","away_indoor_to_outdoor"],
      "recent_exposure":["home_recent_cold_shock","away_recent_cold_shock","home_recent_heat_shock","away_recent_heat_shock","away_low_outdoor_exposure"],
      "threshold_shocks":["away_warm_team_freeze","away_cold_team_heat","away_indoor_to_outdoor"],
      "wind_plus_recent":["wind_over_10","wind_over_15","home_recent_cold_shock","away_recent_cold_shock","home_recent_heat_shock","away_recent_heat_shock","away_low_outdoor_exposure"],
      "wind_plus_threshold":["wind_over_10","wind_over_15","away_warm_team_freeze","away_cold_team_heat","away_indoor_to_outdoor"],
    }
    rows=[]
    for name,features in sets.items(): rows.append({"model":name,**_metrics(holdout.actual_total,_fit(train,holdout,features))})
    print(pd.DataFrame(rows).sort_values(["mae","rmse"]).to_string(index=False)); _threshold_report(games)
if __name__=="__main__": run()
