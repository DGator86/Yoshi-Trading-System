import { useState, useEffect } from 'react'
import axios from 'axios'
import MetricCards from './components/MetricCards'
import PerformanceCharts from './components/PerformanceCharts'
import AccuracyAnalysis from './components/AccuracyAnalysis'
import PredictionTable from './components/PredictionTable'
import Header from './components/Header'
import './App.css'

export default function App() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [lastUpdate, setLastUpdate] = useState(new Date())

  const API_BASE = process.env.VITE_API_BASE || 'http://localhost:8001/api'

  const fetchDashboardData = async () => {
    try {
      setLoading(true)
      const response = await axios.get(`${API_BASE}/dashboard`, {
        timeout: 10000
      })
      setData(response.data)
      setLastUpdate(new Date())
      setError(null)
    } catch (err) {
      setError(err.message || 'Failed to load dashboard data')
      console.error('Dashboard error:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    // Initial load
    fetchDashboardData()

    // Refresh every 30 seconds
    const interval = setInterval(fetchDashboardData, 30000)

    return () => clearInterval(interval)
  }, [])

  if (error) {
    return (
      <div className="min-h-screen bg-slate-900 text-white p-8 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-red-400 mb-4">Connection Error</h1>
          <p className="text-xl text-slate-400 mb-8">{error}</p>
          <p className="text-slate-500 mb-4">Make sure the tracking server is running:</p>
          <code className="bg-slate-800 p-4 rounded text-cyan-400 block mb-6">
            python scripts/tracking_server.py --run-dir ./run_results
          </code>
          <button
            onClick={fetchDashboardData}
            className="bg-cyan-600 hover:bg-cyan-700 text-white font-bold py-2 px-6 rounded"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-900 text-white p-8 flex items-center justify-center">
        <div className="text-center">
          <div className="mb-4 text-6xl">🦖</div>
          <h1 className="text-3xl font-bold mb-4">Loading Yoshi Dashboard...</h1>
          <div className="inline-block animate-spin h-8 w-8 border-4 border-cyan-500 border-t-transparent rounded-full"></div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-slate-900 text-white">
      <Header lastUpdate={lastUpdate} onRefresh={fetchDashboardData} />

      <main className="max-w-7xl mx-auto px-4 py-8 space-y-8">
        {/* Key Performance Indicators */}
        <section>
          <div className="section-header">
            📊 Model Effectiveness & Trading Performance
          </div>
          {data && <MetricCards stats={data.trade_stats} />}
        </section>

        {/* ML Statistical Progress */}
        <section>
          <div className="section-header">
            📈 ML Statistical Progress & Calibration
          </div>
          {data && <PerformanceCharts data={data} />}
        </section>

        {/* Price Prediction Validation */}
        <section>
          <div className="section-header">
            🔮 Price Prediction Validation & Accuracy
          </div>
          {data && <AccuracyAnalysis data={data} />}
        </section>

        {/* Real-time Predictions */}
        <section>
          <div className="section-header">
            ⚡ Latest Predictions
          </div>
          {data && <PredictionTable predictions={data.latest_predictions} />}
        </section>

        {/* Statistics Summary */}
        <section className="bg-slate-800 border border-slate-700 rounded-lg p-8">
          <h2 className="text-xl font-bold text-cyan-400 mb-6">ML Statistical Summary</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold text-slate-300 mb-4">Prediction Accuracy</h3>
              <div className="space-y-2 text-sm">
                <p>📊 Total Predictions: <span className="text-cyan-400 font-mono">{data.prediction_stats.total_predictions}</span></p>
                <p>✓ Directional Accuracy: <span className="text-green-400 font-mono">{(data.prediction_stats.directional_accuracy * 100).toFixed(2)}%</span></p>
                <p>📏 MAE: <span className="text-slate-300 font-mono">{data.prediction_stats.mae.toFixed(4)}</span></p>
                <p>📉 RMSE: <span className="text-slate-300 font-mono">{data.prediction_stats.rmse.toFixed(4)}</span></p>
              </div>
            </div>
            <div>
              <h3 className="font-semibold text-slate-300 mb-4">Calibration & Coverage</h3>
              <div className="space-y-2 text-sm">
                <p>⚖️ Calibration Error (ECE): <span className="text-slate-300 font-mono">{data.prediction_stats.calibration_error.toFixed(4)}</span></p>
                <p>📦 Coverage 90%: <span className="text-slate-300 font-mono">{(data.prediction_stats.coverage_90 * 100).toFixed(2)}%</span></p>
                <p>📦 Coverage 80%: <span className="text-slate-300 font-mono">{(data.prediction_stats.coverage_80 * 100).toFixed(2)}%</span></p>
                <p>⏭️ Abstention Rate: <span className="text-slate-300 font-mono">{(data.prediction_stats.abstention_rate * 100).toFixed(2)}%</span></p>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="bg-slate-800 border-t border-slate-700 mt-12 py-6 text-center text-slate-500 text-sm">
        <p>🦖 Yoshi Trading System - Real-time Prediction Tracking Dashboard</p>
        <p>Last updated: {lastUpdate.toLocaleTimeString()}</p>
      </footer>
    </div>
  )
}
