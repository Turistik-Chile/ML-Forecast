import { motion } from "framer-motion";
import { Button } from "../components/ui/button";
import { Link } from "react-router-dom";

const HomePage = () => {
  return (
    <motion.section
      className="space-y-6 rounded-3xl border border-border bg-background/80 p-10 text-center shadow-xl backdrop-blur"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
    >
      <p className="text-sm font-semibold uppercase tracking-widest text-primary">
        Predicciones con amor
      </p>
      <h1 className="text-4xl font-bold text-foreground">
        Bienvenido a PrediBúnker
      </h1>
      <p className="text-lg text-muted-foreground">
        Explora tu bunker de datos y experimenta con pipelines y modelos para anticipar el futuro.
      </p>
      <div className="flex justify-center gap-4">
        <Button asChild>
          <Link to="/predicciones">Generar predicciones</Link>
        </Button>
      </div>
    </motion.section>
  );
};

export default HomePage;
