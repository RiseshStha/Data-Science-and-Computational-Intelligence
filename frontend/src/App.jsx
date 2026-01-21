import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SearchPage from './components/SearchPage';
import CrawlPage from './components/CrawlPage';
import StatsPage from './components/StatsPage';
import ClusteringPage from './components/ClusteringPage'; // Task 2

function App() {
  return (
    <Router>
      <div className="bg-slate-50 min-h-screen font-sans text-slate-900">
        <Routes>
          <Route path="/" element={<SearchPage />} />
          <Route path="/clustering" element={<ClusteringPage />} /> {/* Task 2 */}
          <Route path="/crawl" element={<CrawlPage />} />
          <Route path="/stats" element={<StatsPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;

