import AnalysisProvider from "./components/AnalysisProvider";
import HeroSection from "./components/HeroSection";
import ResultsSection from "./components/ResultsSection";
import HowItWorks from "./components/HowItWorks";

export default function Home() {
  return (
    <AnalysisProvider>
      <main className="flex min-h-screen flex-col">
        <HeroSection />
        <ResultsSection />
        <div className="flex-1" />
        <HowItWorks />
      </main>
    </AnalysisProvider>
  );
}
