import { NavLink, Link } from "react-router-dom";
import { FiMoon, FiSun } from "react-icons/fi";
import { Button } from "./ui/button";
import { cn } from "../lib/utils";

type NavItem = {
  label: string;
  to: string;
};

type NavbarProps = {
  theme: "light" | "dark";
  onToggleTheme: () => void;
  items: NavItem[];
};

const Navbar = ({ theme, onToggleTheme, items }: NavbarProps) => {
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    cn(
      "text-sm font-semibold transition-colors hover:text-primary",
      isActive ? "text-foreground" : "text-muted-foreground"
    );

  return (
    <header className="sticky top-0 z-30 border-b border-border bg-background/80 backdrop-blur">
      <div className="container flex max-w-5xl items-center justify-between py-4">
        <Link to="/" className="text-lg font-bold tracking-tight text-primary">
          PrediBúnker
        </Link>

        <nav className="flex items-center gap-6">
          {items.map((item) => (
            <NavLink key={item.to} to={item.to} className={linkClass}>
              {item.label}
            </NavLink>
          ))}

          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label="Cambiar tema"
            onClick={onToggleTheme}
          >
            {theme === "dark" ? (
              <FiSun className="h-4 w-4" />
            ) : (
              <FiMoon className="h-4 w-4" />
            )}
          </Button>
        </nav>
      </div>
    </header>
  );
};

export default Navbar;
