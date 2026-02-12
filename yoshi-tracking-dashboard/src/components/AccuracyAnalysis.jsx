export default function AccuracyAnalysis({ data }) {
  const { accuracy_by_confidence = {}, accuracy_decay = {} } = data

  const confidenceBuckets = Object.entries(accuracy_by_confidence).sort()
  const timeHorizons = Object.entries(accuracy_decay).sort()

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Accuracy by Confidence */}
      <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
        <h3 className="font-bold text-cyan-400 mb-6">📊 Accuracy by Confidence Level</h3>
        {confidenceBuckets.length > 0 ? (
          <div className="space-y-3">
            {confidenceBuckets.map(([bucket, accuracy]) => (
              <div key={bucket}>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm text-slate-300">{bucket}</span>
                  <span className={`text-sm font-bold ${accuracy >= 0.5 ? 'text-green-400' : 'text-yellow-400'}`}>
                    {(accuracy * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-700 rounded-full h-3">
                  <div
                    className={`h-3 rounded-full transition-all ${accuracy >= 0.5 ? 'bg-green-500' : 'bg-yellow-500'}`}
                    style={{ width: `${accuracy * 100}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-slate-500">No confidence data available</p>
        )}
        <div className="mt-6 pt-4 border-t border-slate-700 text-xs text-slate-400">
          <p>✓ Higher confidence should correlate with higher accuracy</p>
          <p>⚠ Flat or inverted curves indicate calibration issues</p>
        </div>
      </div>

      {/* Accuracy Decay Over Horizon */}
      <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
        <h3 className="font-bold text-cyan-400 mb-6">📉 Prediction Degradation Over Time Horizon</h3>
        {timeHorizons.length > 0 ? (
          <div className="space-y-3">
            {timeHorizons.map(([horizon, accuracy]) => {
              const hoursAhead = parseInt(horizon) * 4 // Assuming 15min bars
              return (
                <div key={horizon}>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-sm text-slate-300">
                      {hoursAhead}h ahead ({horizon})
                    </span>
                    <span className={`text-sm font-bold ${accuracy >= 0.5 ? 'text-green-400' : 'text-orange-400'}`}>
                      {(accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-700 rounded-full h-3">
                    <div
                      className={`h-3 rounded-full transition-all ${accuracy >= 0.5 ? 'bg-green-500' : 'bg-orange-500'}`}
                      style={{ width: `${accuracy * 100}%` }}
                    ></div>
                  </div>
                </div>
              )
            })}
          </div>
        ) : (
          <p className="text-slate-500">No time horizon data available</p>
        )}
        <div className="mt-6 pt-4 border-t border-slate-700 text-xs text-slate-400">
          <p>✓ Time-agnostic prediction is the goal</p>
          <p>⚠ Steep decay = model struggles with longer horizons</p>
        </div>
      </div>

      {/* Time-Agnostic Performance Summary */}
      <div className="lg:col-span-2 bg-slate-800 border border-slate-700 rounded-lg p-6">
        <h3 className="font-bold text-cyan-400 mb-4">🎯 Time-Agnostic Prediction Quality</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-slate-700/50 rounded p-4">
            <div className="text-sm text-slate-400 mb-2">Confidence Consistency</div>
            <div className="text-lg font-bold">
              {confidenceBuckets.length > 0 ? (
                <>
                  <span className="text-cyan-400">
                    {confidenceBuckets.length}
                  </span>
                  <span className="text-xs text-slate-400 ml-2">bins</span>
                </>
              ) : (
                <span className="text-slate-500">N/A</span>
              )}
            </div>
            <p className="text-xs text-slate-500 mt-2">Higher = more granular confidence levels</p>
          </div>

          <div className="bg-slate-700/50 rounded p-4">
            <div className="text-sm text-slate-400 mb-2">Time Horizon Coverage</div>
            <div className="text-lg font-bold">
              {timeHorizons.length > 0 ? (
                <>
                  <span className="text-cyan-400">
                    {timeHorizons.length}
                  </span>
                  <span className="text-xs text-slate-400 ml-2">horizons</span>
                </>
              ) : (
                <span className="text-slate-500">N/A</span>
              )}
            </div>
            <p className="text-xs text-slate-500 mt-2">Coverage across multiple timeframes</p>
          </div>

          <div className="bg-slate-700/50 rounded p-4">
            <div className="text-sm text-slate-400 mb-2">Prediction Quality</div>
            <div className="text-lg font-bold">
              {confidenceBuckets.length > 0 && timeHorizons.length > 0 ? (
                <>
                  <span className="text-green-400">Robust</span>
                </>
              ) : (
                <span className="text-yellow-400">Limited</span>
              )}
            </div>
            <p className="text-xs text-slate-500 mt-2">Overall assessment across metrics</p>
          </div>
        </div>
      </div>
    </div>
  )
}
