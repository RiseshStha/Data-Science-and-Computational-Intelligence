import React, { useState, useEffect } from 'react';
import { BarChart3, Activity, Database, Users, Calendar, ArrowLeft, TrendingUp, Zap } from 'lucide-react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';

const StatsPage = () => {
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchStats();
    }, []);

    const fetchStats = async () => {
        try {
            const res = await axios.get('http://127.0.0.1:8000/api/stats/');
            setStats(res.data);
        } catch (err) {
            console.error("Failed to load stats", err);
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-white flex items-center justify-center">
                <div className="animate-pulse flex flex-col items-center">
                    <Activity className="text-primary h-12 w-12 mb-4 animate-spin" />
                    <p className="text-slate-500">Loading Statistics...</p>
                </div>
            </div>
        );
    }

    if (!stats) return null;

    return (
        <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white py-12 px-4 sm:px-6 lg:px-8">
            {/* Navigation */}
            <div className="absolute top-4 left-4 z-20">
                <Link
                    to="/"
                    className="flex items-center space-x-2 text-sm text-slate-500 hover:text-primary transition-colors bg-white/50 backdrop-blur-sm px-3 py-1.5 rounded-full border border-gray-200 hover:border-primary/30 hover:shadow-sm"
                >
                    <ArrowLeft size={16} />
                    <span className="font-medium">Back to Search</span>
                </Link>
            </div>

            <div className="absolute top-4 right-4 z-20">
                <Link
                    to="/crawl"
                    className="flex items-center space-x-2 text-sm text-slate-500 hover:text-primary transition-colors bg-white/50 backdrop-blur-sm px-3 py-1.5 rounded-full border border-gray-200 hover:border-primary/30 hover:shadow-sm"
                >
                    <span className="font-medium">Data Console</span>
                </Link>
            </div>

            <div className="max-w-6xl mx-auto">
                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="text-center mb-12"
                >
                    <div className="flex items-center justify-center space-x-3 mb-4">
                        <BarChart3 size={48} className="text-primary" />
                        <h1 className="text-4xl sm:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-blue-600">
                            Search Engine Statistics
                        </h1>
                    </div>
                    <p className="text-slate-500 text-lg">Performance metrics and index analytics</p>
                </motion.div>

                {/* Key Metrics Cards */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 }}
                        className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-all"
                    >
                        <div className="flex items-center justify-between mb-2">
                            <Database className="h-8 w-8 text-blue-500" />
                            <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">Publications</span>
                        </div>
                        <div className="text-3xl font-bold text-slate-900">{stats.total_publications}</div>
                        <div className="text-xs text-slate-500 mt-1">Total indexed</div>
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2 }}
                        className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-all"
                    >
                        <div className="flex items-center justify-between mb-2">
                            <Zap className="h-8 w-8 text-yellow-500" />
                            <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">Terms</span>
                        </div>
                        <div className="text-3xl font-bold text-slate-900">{stats.unique_terms}</div>
                        <div className="text-xs text-slate-500 mt-1">Unique tokens</div>
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3 }}
                        className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-all"
                    >
                        <div className="flex items-center justify-between mb-2">
                            <Users className="h-8 w-8 text-purple-500" />
                            <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">Authors</span>
                        </div>
                        <div className="text-3xl font-bold text-slate-900">{stats.total_authors}</div>
                        <div className="text-xs text-slate-500 mt-1">Contributors</div>
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.4 }}
                        className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-all"
                    >
                        <div className="flex items-center justify-between mb-2">
                            <Activity className="h-8 w-8 text-green-500" />
                            <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">Status</span>
                        </div>
                        <div className="text-2xl font-bold text-green-600">{stats.status}</div>
                        <div className="text-xs text-slate-500 mt-1">System health</div>
                    </motion.div>
                </div>

                {/* Main Content Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Field Weights */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.5 }}
                        className="bg-white p-6 rounded-xl shadow-sm border border-gray-100"
                    >
                        <h2 className="text-xl font-semibold text-slate-900 mb-4 flex items-center">
                            <TrendingUp className="mr-2 h-5 w-5 text-primary" />
                            Field Weights
                        </h2>
                        <div className="space-y-4">
                            {Object.entries(stats.weights).map(([field, weight]) => (
                                <div key={field}>
                                    <div className="flex justify-between items-center mb-1">
                                        <span className="text-sm font-medium text-slate-700">{field}</span>
                                        <span className="text-sm font-bold text-primary">{weight}×</span>
                                    </div>
                                    <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                                        <div
                                            className="h-full bg-gradient-to-r from-primary to-blue-400 transition-all"
                                            style={{ width: `${(weight / 3.0) * 100}%` }}
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </motion.div>

                    {/* Publications by Year */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.6 }}
                        className="bg-white p-6 rounded-xl shadow-sm border border-gray-100"
                    >
                        <h2 className="text-xl font-semibold text-slate-900 mb-4 flex items-center">
                            <Calendar className="mr-2 h-5 w-5 text-primary" />
                            Publications by Year
                        </h2>
                        <div className="h-64 overflow-y-auto space-y-2">
                            {Object.entries(stats.years_distribution)
                                .sort(([a], [b]) => {
                                    if (a === 'N/A') return 1;
                                    if (b === 'N/A') return -1;
                                    return parseInt(b) - parseInt(a);
                                })
                                .map(([year, count]) => (
                                    <div key={year} className="flex items-center group">
                                        <span className="w-16 text-sm font-medium text-slate-600">{year}</span>
                                        <div className="flex-1 h-6 bg-slate-50 rounded mx-2 relative overflow-hidden">
                                            <div
                                                className="absolute top-0 left-0 h-full bg-blue-500 group-hover:bg-blue-600 transition-all"
                                                style={{ width: `${Math.min((count / Math.max(...Object.values(stats.years_distribution))) * 100, 100)}%` }}
                                            />
                                        </div>
                                        <span className="w-12 text-right text-sm font-semibold text-slate-700">{count}</span>
                                    </div>
                                ))}
                        </div>
                    </motion.div>

                    {/* Algorithm Info */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.7 }}
                        className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 lg:col-span-2"
                    >
                        <h2 className="text-xl font-semibold text-slate-900 mb-4">Ranking Algorithm</h2>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div className="p-4 bg-blue-50 rounded-lg border border-blue-100">
                                <h3 className="font-semibold text-blue-900 mb-2">TF-IDF Scoring</h3>
                                <p className="text-sm text-blue-700">
                                    Computes term frequency-inverse document frequency to identify rare, important keywords.
                                </p>
                            </div>
                            <div className="p-4 bg-purple-50 rounded-lg border border-purple-100">
                                <h3 className="font-semibold text-purple-900 mb-2">Field Boosting</h3>
                                <p className="text-sm text-purple-700">
                                    Title matches are weighted 3.0×, ensuring highly relevant documents rank higher.
                                </p>
                            </div>
                            <div className="p-4 bg-green-50 rounded-lg border border-green-100">
                                <h3 className="font-semibold text-green-900 mb-2">Metadata Rich</h3>
                                <p className="text-sm text-green-700">
                                    Index includes abstracts, authors, and publication year for comprehensive coverage.
                                </p>
                            </div>
                        </div>
                    </motion.div>

                    {/* System Info */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.8 }}
                        className="bg-white p-6 rounded-xl shadow-sm border border-gray-100"
                    >
                        <h2 className="text-xl font-semibold text-slate-900 mb-4">System Information</h2>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                            <div>
                                <span className="text-slate-500">Last Updated:</span>
                                <p className="font-medium text-slate-900 mt-1">
                                    {stats.last_updated ? new Date(stats.last_updated).toLocaleString() : 'Never'}
                                </p>
                            </div>
                            <div>
                                <span className="text-slate-500">Database:</span>
                                <p className="font-medium text-slate-900 mt-1">SQLite (db.sqlite3)</p>
                            </div>
                            <div>
                                <span className="text-slate-500">Storage:</span>
                                <p className="font-medium text-slate-900 mt-1">Local Filesystem</p>
                            </div>
                        </div>
                    </motion.div>

                    {/* Performance Benchmarks */}
                    {stats.benchmarks && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.9 }}
                            className="bg-gradient-to-br from-green-50 to-emerald-50 p-6 rounded-xl shadow-sm border border-green-200"
                        >
                            <h2 className="text-xl font-semibold text-green-900 mb-4 flex items-center">
                                <Zap className="mr-2 h-5 w-5 text-green-600" />
                                Performance Benchmarks
                            </h2>
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                <div className="bg-white p-4 rounded-lg border border-green-100">
                                    <div className="text-xs text-green-600 font-semibold uppercase tracking-wider mb-1">
                                        Avg Search Time
                                    </div>
                                    <div className="text-2xl font-bold text-green-900">
                                        {stats.benchmarks.avg_search_time_ms}
                                        <span className="text-sm font-normal text-green-600 ml-1">ms</span>
                                    </div>
                                    <div className="text-xs text-green-700 mt-1">
                                        Range: {stats.benchmarks.min_search_time_ms} - {stats.benchmarks.max_search_time_ms} ms
                                    </div>
                                </div>

                                <div className="bg-white p-4 rounded-lg border border-green-100">
                                    <div className="text-xs text-green-600 font-semibold uppercase tracking-wider mb-1">
                                        Index Efficiency
                                    </div>
                                    <div className="text-2xl font-bold text-green-900">
                                        {stats.benchmarks.index_density}
                                        <span className="text-sm font-normal text-green-600 ml-1">terms/doc</span>
                                    </div>
                                    <div className="text-xs text-green-700 mt-1">
                                        {stats.benchmarks.docs_per_term} docs per term
                                    </div>
                                </div>

                                <div className="bg-white p-4 rounded-lg border border-green-100">
                                    <div className="text-xs text-green-600 font-semibold uppercase tracking-wider mb-1">
                                        Est. Index Build
                                    </div>
                                    <div className="text-2xl font-bold text-green-900">
                                        {(stats.benchmarks.estimated_full_index_time_ms / 1000).toFixed(2)}
                                        <span className="text-sm font-normal text-green-600 ml-1">sec</span>
                                    </div>
                                    <div className="text-xs text-green-700 mt-1">
                                        For {stats.total_publications} documents
                                    </div>
                                </div>
                            </div>
                            <div className="mt-4 p-3 bg-white/50 rounded-lg border border-green-100">
                                <p className="text-xs text-green-800">
                                    <strong>Note:</strong> Benchmarks measured using 5 test queries.
                                    Search times under 10ms indicate excellent performance for real-time search.
                                </p>
                            </div>
                        </motion.div>
                    )}
                </div>
            </div>

            {/* Footer */}
            <div className="fixed bottom-0 w-full py-4 text-center text-xs text-gray-400 bg-slate-50/80 backdrop-blur-sm">
                <p>Research Publication Search • Powered by Django & React</p>
            </div>
        </div>
    );
};

export default StatsPage;
