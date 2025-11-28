import { Card, CardDescription, CardHeader, CardTitle } from "./ui/card";

type PageHeaderProps = {
  title: string;
  subtitle?: string;
};

const PageHeader = ({ title, subtitle }: PageHeaderProps) => (
  <Card className="border border-border bg-background/90 shadow-xl shadow-ring/10 backdrop-blur-md transition-colors">
    <CardHeader className="space-y-2">
      <CardTitle className="text-3xl tracking-tight text-foreground">{title}</CardTitle>
      {subtitle ? (
        <CardDescription className="text-base text-muted-foreground">{subtitle}</CardDescription>
      ) : null}
    </CardHeader>
  </Card>
);

export default PageHeader;
