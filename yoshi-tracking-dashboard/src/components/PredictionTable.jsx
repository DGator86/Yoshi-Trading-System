export default function PredictionTable({ predictions }) {
  const formatPrice = (price) => `$${price.toFixed(2)}`
  const formatPercent = (pct) => `${(pct * 100).toFixed(2)}%`

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-700">
            <th className="text-left py-3 px-4 text-slate-300">Symbol</th>
            <th className="text-left py-3 px-4 text-slate-300">Timestamp</th>
            <th className="text-right py-3 px-4 text-slate-300">Price</th>
            <th className="text-right py-3 px-4 text-slate-300">Pred (Q50)</th>
            <th className="text-right py-3 px-4 text-slate-300">CI 90%</th>
            <th className="text-right py-3 px-4 text-slate-300">Confidence</th>
            <th className="text-right py-3 px-4 text-slate-300">Actual Return</th>
            <th className="text-center py-3 px-4 text-slate-300">Result</th>
          </tr>
        </thead>
        <tbody>
          {predictions.map((pred, idx) => (
            <tr
              key={idx}
              className={`border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors ${
                pred.correct ? 'prediction-correct' : 'prediction-incorrect'
              }`}
            >
              <td className="py-3 px-4 font-mono font-bold text-cyan-400">{pred.symbol}</td>
              <td className="py-3 px-4 text-slate-400 text-xs">
                {new Date(pred.timestamp).toLocaleString()}
              </td>
              <td className="py-3 px-4 text-right text-slate-300">{formatPrice(pred.actual_price)}</td>
              <td className="py-3 px-4 text-right font-bold text-cyan-400">
                {formatPrice(pred.predicted_median)}
              </td>
              <td className="py-3 px-4 text-right text-slate-400 text-xs">
                {formatPrice(pred.predicted_q05)} - {formatPrice(pred.predicted_q95)}
              </td>
              <td className="py-3 px-4 text-right">
                <div className="flex items-center justify-end gap-2">
                  <div className="w-16 bg-slate-700 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-yellow-500 to-green-500 h-2 rounded-full"
                      style={{ width: `${Math.min(pred.confidence * 100, 100)}%` }}
                    ></div>
                  </div>
                  <span className="text-slate-300 text-xs font-mono">{(pred.confidence * 100).toFixed(0)}%</span>
                </div>
              </td>
              <td className={`py-3 px-4 text-right font-bold ${
                pred.actual_return > 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {formatPercent(pred.actual_return)}
              </td>
              <td className="py-3 px-4 text-center">
                {pred.correct ? (
                  <span className="inline-block px-3 py-1 bg-green-900/40 text-green-400 rounded text-xs font-bold">
                    ✓ HIT
                  </span>
                ) : (
                  <span className="inline-block px-3 py-1 bg-red-900/40 text-red-400 rounded text-xs font-bold">
                    ✗ MISS
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {predictions.length === 0 && (
        <div className="text-center py-8 text-slate-500">
          No predictions available
        </div>
      )}

      {predictions.length > 0 && (
        <div className="mt-4 pt-4 border-t border-slate-700 flex justify-between text-sm text-slate-400">
          <div>
            📊 Showing {predictions.length} most recent predictions
          </div>
          <div>
            ✓ Hit Rate: <span className="text-green-400 font-bold">
              {(predictions.filter(p => p.correct).length / predictions.length * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
