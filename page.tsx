import { useEffect, useState } from "react";
import KisanBot from "@/components/KisanBot";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { RefreshCw, Sprout, Leaf, Droplets, Upload, Scan, Loader2 } from "lucide-react";
import { api, type Sensors } from "@/lib/api";
import { toast } from "sonner";

const SensorTile = ({ label, value, unit }: { label: string; value: string; unit?: string }) => (
  <Card className="p-3 text-center">
    <p className="text-xs text-muted-foreground">{label}</p>
    <p className="text-lg font-bold text-primary mt-1">
      {value}
      {unit && <span className="text-sm font-normal ml-0.5">{unit}</span>}
    </p>
  </Card>
);

const Index = () => {
  const [soil, setSoil] = useState({ N: "", P: "", K: "", ph: "" });
  const [moisture, setMoisture] = useState("");
  const [sensors, setSensors] = useState<Sensors | null>(null);
  const [sensorStatus, setSensorStatus] = useState<"waiting" | "live" | "error">("waiting");

  const [diseaseFile, setDiseaseFile] = useState<File | null>(null);
  const [diseaseResult, setDiseaseResult] = useState<any>(null);
  const [soilResult, setSoilResult] = useState<any>(null);
  const [cropResult, setCropResult] = useState<string | null>(null);
  const [irrigResult, setIrrigResult] = useState<any>(null);

  const [loading, setLoading] = useState<{ [k: string]: boolean }>({});
  const setLoad = (k: string, v: boolean) => setLoading((p) => ({ ...p, [k]: v }));

  const fetchSensors = async (manual = false) => {
    try {
      const data = await api.getSensors();
      setSensors(data);
      const isLive = !!(data.moisture || data.temp || data.N || data.P || data.K || data.ph);
      setSensorStatus(isLive ? "live" : "waiting");
      setSoil((prev) => ({
        N: prev.N || (data.N ? String(data.N) : ""),
        P: prev.P || (data.P ? String(data.P) : ""),
        K: prev.K || (data.K ? String(data.K) : ""),
        ph: prev.ph || (data.ph ? String(data.ph) : ""),
      }));
      setMoisture((prev) => prev || (data.moisture ? String(data.moisture) : ""));
      if (manual) {
        toast.success(isLive ? "सेंसर अपडेट हुए / Sensors updated" : "कनेक्टेड — सेंसर डेटा शून्य है / Connected — sensor data is 0");
      }
    } catch (e: any) {
      console.error("Sensor fetch failed:", e);
      setSensorStatus("error");
      if (manual) toast.error(`Sensor fetch failed: ${e.message}`);
    }
  };

  useEffect(() => {
    fetchSensors();
    const id = setInterval(fetchSensors, 5000);
    return () => clearInterval(id);
  }, []);

  const handleAnalyzeSoil = async () => {
    const payload = { N: +soil.N, P: +soil.P, K: +soil.K, ph: +soil.ph };
    if ([payload.N, payload.P, payload.K, payload.ph].some((v) => isNaN(v))) {
      toast.error("कृपया सभी मान भरें / Fill all fields");
      return;
    }
    setLoad("soil", true);
    try {
      const res = await api.analyzeSoil(payload);
      setSoilResult(res);
      toast.success(res.fertility_level);
    } catch (e: any) {
      toast.error(e.message);
    } finally {
      setLoad("soil", false);
    }
  };

  const handleCropSuggest = async () => {
    const payload = { N: +soil.N, P: +soil.P, K: +soil.K, ph: +soil.ph };
    if ([payload.N, payload.P, payload.K, payload.ph].some((v) => isNaN(v))) {
      toast.error("कृपया सभी मान भरें / Fill all fields");
      return;
    }
    setLoad("crop", true);
    try {
      const res = await api.predictCrop(payload);
      setCropResult(res.crop);
      toast.success(`Suggested: ${res.crop}`);
    } catch (e: any) {
      toast.error(e.message);
    } finally {
      setLoad("crop", false);
    }
  };

  const handleDiseaseScan = async () => {
    if (!diseaseFile) {
      toast.error("पहले तस्वीर चुनें / Select an image first");
      return;
    }
    setLoad("disease", true);
    try {
      const res = await api.predictDisease(diseaseFile);
      setDiseaseResult(res);
    } catch (e: any) {
      toast.error(e.message);
    } finally {
      setLoad("disease", false);
    }
  };

  const handleIrrigation = async () => {
    const m = +moisture;
    if (isNaN(m)) {
      toast.error("Enter moisture %");
      return;
    }
    setLoad("irrig", true);
    try {
      const res = await api.predictIrrigation(m);
      setIrrigResult(res);
    } catch (e: any) {
      toast.error(e.message);
    } finally {
      setLoad("irrig", false);
    }
  };

  const sensorBadge =
    sensorStatus === "live"
      ? { text: "🟢 लाइव सेंसर", cls: "bg-green-500/20" }
      : sensorStatus === "error"
      ? { text: "❌ कनेक्शन त्रुटि", cls: "bg-destructive/30" }
      : { text: "⏳ सेंसर प्रतीक्षा", cls: "bg-primary-foreground/20" };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="bg-primary text-primary-foreground px-4 sm:px-6 py-4 shadow-md">
        <div className="max-w-7xl mx-auto flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-primary-foreground/20 flex items-center justify-center">
              <Sprout className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-xl font-bold leading-tight">Kisan AI Pro</h1>
              <p className="text-xs opacity-80">स्मार्ट खेती · Gemini किसान मित्र</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className={`text-xs px-2 py-1 rounded-full ${sensorBadge.cls}`}>
              {sensorBadge.text}
            </span>
            <Button size="sm" variant="secondary" className="gap-1" onClick={() => fetchSensors(true)}>
              <RefreshCw className="w-3.5 h-3.5" /> Refresh
            </Button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto p-4 sm:p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* LEFT: Sensors + Tabs */}
        <section className="lg:col-span-2 space-y-6">
          {/* Sensor grid */}
          <div className="grid grid-cols-3 sm:grid-cols-6 gap-2">
            <SensorTile label="Moisture" value={sensors ? String(sensors.moisture) : "--"} unit="%" />
            <SensorTile label="Temp" value={sensors ? String(sensors.temp) : "--"} unit="°C" />
            <SensorTile label="N (kg/ha)" value={sensors ? String(sensors.N) : "--"} />
            <SensorTile label="P" value={sensors ? String(sensors.P) : "--"} />
            <SensorTile label="K" value={sensors ? String(sensors.K) : "--"} />
            <SensorTile label="pH" value={sensors ? String(sensors.ph) : "--"} />
          </div>

          {/* Tabs */}
          <Card className="p-4">
            <Tabs defaultValue="soil">
              <TabsList className="grid grid-cols-3 w-full">
                <TabsTrigger value="soil" className="gap-1.5">
                  <Sprout className="w-4 h-4" /> मिट्टी & फसल
                </TabsTrigger>
                <TabsTrigger value="disease" className="gap-1.5">
                  <Leaf className="w-4 h-4" /> रोग निदान
                </TabsTrigger>
                <TabsTrigger value="irrigation" className="gap-1.5">
                  <Droplets className="w-4 h-4" /> सिंचाई
                </TabsTrigger>
              </TabsList>

              {/* Soil tab */}
              <TabsContent value="soil" className="space-y-4 pt-4">
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>📡 सेंसर डेटा स्वतः भरा गया | आप बदल सकते हैं</span>
                  <button className="text-primary font-medium" onClick={() => fetchSensors(true)}>
                    🔄 अपडेट करें
                  </button>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  {(["N", "P", "K", "ph"] as const).map((k) => (
                    <div key={k}>
                      <Label className="text-xs">{k === "ph" ? "pH" : `${k} (mg/kg)`}</Label>
                      <Input
                        type="number"
                        value={soil[k]}
                        onChange={(e) => setSoil({ ...soil, [k]: e.target.value })}
                        placeholder="--"
                      />
                    </div>
                  ))}
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button className="gap-1.5" onClick={handleAnalyzeSoil} disabled={loading.soil}>
                    {loading.soil ? <Loader2 className="w-4 h-4 animate-spin" /> : "🧪"} उर्वरता विश्लेषण
                  </Button>
                  <Button
                    variant="secondary"
                    className="gap-1.5"
                    onClick={handleCropSuggest}
                    disabled={loading.crop}
                  >
                    {loading.crop ? <Loader2 className="w-4 h-4 animate-spin" /> : "🌾"} सर्वोत्तम फसल सुझाव
                  </Button>
                </div>
                {soilResult && (
                  <Card className="p-3 bg-secondary/40 text-sm space-y-1">
                    <p><b>Score:</b> {soilResult.fertility_score}</p>
                    <p><b>Level:</b> {soilResult.fertility_level}</p>
                    <p><b>सिफारिश:</b> {soilResult.recommendation}</p>
                  </Card>
                )}
                {cropResult && (
                  <Card className="p-3 bg-secondary/40 text-sm">
                    🌾 <b>Recommended crop:</b> {cropResult}
                  </Card>
                )}
              </TabsContent>

              {/* Disease tab */}
              <TabsContent value="disease" className="space-y-4 pt-4">
                <label className="border-2 border-dashed border-border rounded-lg p-8 flex flex-col items-center gap-2 cursor-pointer hover:border-primary hover:bg-secondary/50 transition-colors">
                  <Upload className="w-8 h-8 text-muted-foreground" />
                  <p className="text-sm text-muted-foreground">
                    {diseaseFile ? diseaseFile.name : "पत्ती की तस्वीर लें या अपलोड करें"}
                  </p>
                  <input
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => setDiseaseFile(e.target.files?.[0] ?? null)}
                  />
                </label>
                <Button className="w-full gap-1.5" onClick={handleDiseaseScan} disabled={loading.disease}>
                  {loading.disease ? <Loader2 className="w-4 h-4 animate-spin" /> : <Scan className="w-4 h-4" />}
                  AI स्कैन शुरू करें
                </Button>
                {diseaseResult && (
                  <Card className="p-3 bg-secondary/40 text-sm space-y-1">
                    <p>
                      <b>रोग:</b> {diseaseResult.name} ({(() => {
                        const confidence = Number(diseaseResult.confidence);
                        const normalized = confidence > 0 && confidence < 1 ? confidence * 100 : confidence;
                        return `${normalized.toFixed(2)}%`;
                      })()})
                    </p>
                    <p><b>Severity:</b> {diseaseResult.severity}</p>
                    <p><b>Pesticide:</b> {diseaseResult.pesticide} — {diseaseResult.dosage}</p>
                    <p><b>Schedule:</b> {diseaseResult.schedule}</p>
                    <p><b>Organic:</b> {diseaseResult.organic}</p>
                    <p><b>उपचार:</b> {diseaseResult.cure}</p>
                  </Card>
                )}
              </TabsContent>

              {/* Irrigation tab */}
              <TabsContent value="irrigation" className="space-y-4 pt-4">
                <div>
                  <Label className="text-xs">नमी (Soil Moisture %)</Label>
                  <Input
                    type="number"
                    value={moisture}
                    onChange={(e) => setMoisture(e.target.value)}
                    placeholder="0-100"
                  />
                </div>
                <Button className="w-full gap-1.5" onClick={handleIrrigation} disabled={loading.irrig}>
                  {loading.irrig ? <Loader2 className="w-4 h-4 animate-spin" /> : <Droplets className="w-4 h-4" />}
                  सिंचाई भविष्यवाणी
                </Button>
                {irrigResult && (
                  <Card className="p-3 bg-secondary/40 text-sm space-y-1">
                    <p className="text-base font-semibold">{irrigResult.decision}</p>
                    <p><b>Status:</b> {irrigResult.status}</p>
                    <p><b>Confidence:</b> {irrigResult.confidence}%</p>
                  </Card>
                )}
              </TabsContent>
            </Tabs>
          </Card>
        </section>

        {/* RIGHT: KisanBot (untouched) */}
        <aside className="lg:col-span-1">
          <div className="h-[600px] lg:h-[calc(100vh-8rem)] lg:sticky lg:top-6">
            <KisanBot />
          </div>
        </aside>
      </main>
    </div>
  );
};

export default Index;
