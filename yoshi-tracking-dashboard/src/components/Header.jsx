export default function Header({ lastUpdate, onRefresh }) {
  return (
    <header className="bg-gradient-to-r from-slate-800 to-slate-900 border-b border-slate-700 py-8 shadow-lg">
      <div className="max-w-7xl mx-auto px-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="text-5xl">🦖</div>
          <div>
            <h1 className="text-4xl font-bold text-white">Yoshi Prediction Tracker</h1>
            <p className="text-cyan-400 text-sm mt-1">Real-time ML Performance & Price Prediction Dashboard</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right text-sm text-slate-400">
            <div>Last Update:</div>
            <div className="text-cyan-400 font-mono">{lastUpdate.toLocaleTimeString()}</div>
          </div>
          <button
            onClick={onRefresh}
            className="bg-cyan-600 hover:bg-cyan-700 text-white font-bold py-2 px-4 rounded transition-colors"
            title="Refresh data"
          >
            🔄 Refresh
          </button>
        </div>
      </div>
    </header>
  )
}
