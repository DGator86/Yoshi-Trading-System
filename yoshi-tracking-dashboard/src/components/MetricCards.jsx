export default function MetricCards({ stats }) {
  const metrics = [
    {
      label: 'Total Return',
      value: stats.total_return,
      format: (v) => `${v.toFixed(2)}%`,
      icon: '📈',
      isPositive: stats.total_return > 0
    },
    {
      label: 'Annualized Return',
      value: stats.annualized_return,
      format: (v) => `${v.toFixed(2)}%`,
      icon: '📊',
      isPositive: stats.annualized_return > 0
    },
    {
      label: 'Sharpe Ratio',
      value: stats.sharpe_ratio,
      format: (v) => v.toFixed(3),
      icon: '⚙️',
      isPositive: stats.sharpe_ratio > 1
    },
    {
      label: 'Win Rate',
      value: stats.win_rate,
      format: (v) => `${(v * 100).toFixed(1)}%`,
      icon: '✓',
      isPositive: stats.win_rate > 0.5
    },
    {
      label: 'Max Drawdown',
      value: stats.max_drawdown,
      format: (v) => `${v.toFixed(2)}%`,
      icon: '📉',
      isPositive: stats.max_drawdown > -20
    },
    {
      label: 'Profit Factor',
      value: stats.profit_factor,
      format: (v) => v.toFixed(3),
      icon: '💰',
      isPositive: stats.profit_factor > 1
    },
    {
      label: 'Trades',
      value: stats.num_trades,
      format: (v) => Math.round(v),
      icon: '🎯',
      isPositive: true
    },
    {
      label: 'Avg Trade Return',
      value: stats.avg_trade_return,
      format: (v) => `${v.toFixed(4)}%`,
      icon: '💵',
      isPositive: stats.avg_trade_return > 0
    }
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {metrics.map((metric, idx) => (
        <div key={idx} className={`metric-card ${metric.isPositive ? 'border-green-700/50' : 'border-red-700/50'}`}>
          <div className="text-3xl mb-2">{metric.icon}</div>
          <div className="text-slate-400 text-sm font-semibold">{metric.label}</div>
          <div className={`text-2xl font-bold mt-2 ${metric.isPositive ? 'text-green-400' : 'text-red-400'}`}>
            {metric.format(metric.value)}
          </div>
        </div>
      ))}
    </div>
  )
}
