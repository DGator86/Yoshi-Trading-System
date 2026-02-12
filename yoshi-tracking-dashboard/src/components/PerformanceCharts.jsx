import { useMemo } from 'react'

export default function PerformanceCharts({ data }) {
  const { prediction_stats, equity_metrics, calibration_curve } = data

  const calibrationData = useMemo(() => {
    if (!calibration_curve?.predicted_probabilities?.length) {
      return { prepared: false }
    }

    return {
      prepared: true,
      predicted: calibration_curve.predicted_probabilities,
      observed: calibration_curve.observed_accuracies
    }
  }, [calibration_curve])

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Prediction Statistics */}
      <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
        <h3 className="font-bold text-cyan-400 mb-6">Prediction Statistics</h3>
        <div className="space-y-4">
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-slate-300">Directional Accuracy</span>
              <span className="text-cyan-400 font-bold">{(prediction_stats.directional_accuracy * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div
                className="bg-cyan-500 h-2 rounded-full transition-all"
                style={{ width: `${Math.min(prediction_stats.directional_accuracy * 100, 100)}%` }}
              ></div>
            </div>
          </div>

          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-slate-300">Coverage at 90% CI</span>
              <span className="text-cyan-400 font-bold">{(prediction_stats.coverage_90 * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div
                className="bg-green-500 h-2 rounded-full transition-all"
                style={{ width: `${Math.min(prediction_stats.coverage_90 * 100, 100)}%` }}
              ></div>
            </div>
          </div>

          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-slate-300">Coverage at 80% CI</span>
              <span className="text-cyan-400 font-bold">{(prediction_stats.coverage_80 * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div
                className="bg-emerald-500 h-2 rounded-full transition-all"
                style={{ width: `${Math.min(prediction_stats.coverage_80 * 100, 100)}%` }}
              ></div>
            </div>
          </div>

          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-slate-300">Calibration Error (ECE)</span>
              <span className="text-slate-300 font-bold">{prediction_stats.calibration_error.toFixed(4)}</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all ${prediction_stats.calibration_error < 0.05 ? 'bg-green-500' : 'bg-yellow-500'}`}
                style={{ width: `${Math.min((1 - prediction_stats.calibration_error) * 100, 100)}%` }}
              ></div>
            </div>
          </div>

          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-slate-300">Sharpness (Interval Width)</span>
              <span className="text-slate-300 font-bold">{prediction_stats.sharpness_mean.toFixed(4)}</span>
            </div>
          </div>

          <div className="pt-4 border-t border-slate-700 mt-4">
            <div className="text-sm text-slate-400">
              <p>📊 Total Predictions: <span className="text-cyan-400">{prediction_stats.total_predictions}</span></p>
              <p>⏭️ Abstentions: <span className="text-yellow-400">{(prediction_stats.abstention_rate * 100).toFixed(1)}%</span></p>
              <p>🎯 MAE: <span className="text-slate-300">{prediction_stats.mae.toFixed(6)}</span></p>
              <p>📈 RMSE: <span className="text-slate-300">{prediction_stats.rmse.toFixed(6)}</span></p>
            </div>
          </div>
        </div>
      </div>

      {/* Equity Metrics */}
      <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
        <h3 className="font-bold text-cyan-400 mb-6">Equity Curve Analytics</h3>
        <div className="space-y-4">
          <div className="bg-slate-700/50 rounded p-4">
            <div className="text-sm text-slate-400 mb-1">Final Equity</div>
            <div className="text-3xl font-bold text-green-400">${equity_metrics.final_equity.toFixed(2)}</div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="bg-slate-700/50 rounded p-3">
              <div className="text-xs text-slate-400 mb-1">Max Equity</div>
              <div className="text-lg font-bold text-cyan-400">${equity_metrics.max_equity.toFixed(2)}</div>
            </div>
            <div className="bg-slate-700/50 rounded p-3">
              <div className="text-xs text-slate-400 mb-1">Min Equity</div>
              <div className="text-lg font-bold text-red-400">${equity_metrics.min_equity.toFixed(2)}</div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="bg-slate-700/50 rounded p-3">
              <div className="text-xs text-slate-400 mb-1">Volatility</div>
              <div className="text-lg font-bold text-slate-300">{(equity_metrics.volatility * 100).toFixed(2)}%</div>
            </div>
            <div className="bg-slate-700/50 rounded p-3">
              <div className="text-xs text-slate-400 mb-1">Skewness</div>
              <div className="text-lg font-bold text-slate-300">{equity_metrics.skewness.toFixed(3)}</div>
            </div>
          </div>

          <div className="pt-4 border-t border-slate-700">
            <p className="text-xs text-slate-400">
              🎲 Kurtosis: <span className="text-slate-300 font-mono">{equity_metrics.kurtosis.toFixed(3)}</span>
            </p>
            <p className="text-xs text-slate-500 mt-2">
              {equity_metrics.skewness > 0.5 ? '✓ Positive skew - more large gains' : '⚠ Negative skew - risk of large losses'}
            </p>
          </div>
        </div>
      </div>

      {/* Calibration Curve */}
      {calibrationData.prepared && (
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 lg:col-span-2">
          <h3 className="font-bold text-cyan-400 mb-6">Calibration Curve (Predicted vs Observed)</h3>
          <div className="space-y-3">
            {calibrationData.predicted.map((pred, idx) => {
              const obs = calibrationData.observed[idx] || 0
              const calibrationError = Math.abs(pred - obs)
              const isWellCalibrated = calibrationError < 0.1

              return (
                <div key={idx} className="flex items-center gap-4">
                  <div className="w-12 text-xs font-mono text-slate-400">
                    {(pred * 100).toFixed(0)}%
                  </div>
                  <div className="flex-1 h-6 bg-slate-700 rounded relative overflow-hidden">
                    <div
                      className="h-full bg-cyan-500/50 transition-all"
                      style={{ width: `${pred * 100}%` }}
                      title="Predicted probability"
                    ></div>
                    <div
                      className={`h-full absolute top-0 transition-all ${isWellCalibrated ? 'bg-green-500/50' : 'bg-yellow-500/50'}`}
                      style={{ width: `${obs * 100}%` }}
                      title="Observed accuracy"
                    ></div>
                  </div>
                  <div className="w-12 text-xs font-mono text-slate-300 text-right">
                    {(obs * 100).toFixed(0)}%
                  </div>
                </div>
              )
            })}
          </div>
          <div className="text-xs text-slate-500 mt-4 text-center">
            Cyan = Predicted | Green = Observed (Well-calibrated) | Yellow = Observed (Needs calibration)
          </div>
        </div>
      )}
    </div>
  )
}
