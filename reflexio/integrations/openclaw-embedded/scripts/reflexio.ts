import * as fs from "node:fs";
import * as path from "node:path";
import { writeProfile } from "./lib/write-profile.js";
import { writePlaybook } from "./lib/write-playbook.js";
import { search } from "./lib/search.js";

function loadConfig() {
  const configPath = path.resolve(
    import.meta.dirname || __dirname,
    "..",
    "config.json"
  );
  try {
    return JSON.parse(fs.readFileSync(configPath, "utf8"));
  } catch {
    return {
      dedup: { shallow_threshold: 0.4, top_k: 5 },
    };
  }
}

function parseArgs(args: string[]): Record<string, string> {
  const result: Record<string, string> = {};
  let i = 0;
  while (i < args.length) {
    if (args[i].startsWith("--")) {
      const key = args[i].slice(2);
      const value = args[i + 1] && !args[i + 1].startsWith("--") ? args[i + 1] : "";
      result[key] = value;
      i += value ? 2 : 1;
    } else {
      i++;
    }
  }
  return result;
}

function usage(): never {
  console.error(`Usage: reflexio.ts <command> [options]

Commands:
  write-profile  --slug <s> --ttl <t> --body <text>
  write-playbook --slug <s> --body <text>
  search         --query <text>

Options:
  --slug    kebab-case identifier (e.g. diet-vegan)
  --ttl     one_day | one_week | one_month | one_quarter | one_year | infinity
  --body    content text
  --query   search query text
`);
  process.exit(2);
}

function main() {
  const [command, ...rest] = process.argv.slice(2);
  if (!command) usage();

  const flags = parseArgs(rest);
  const config = loadConfig();

  try {
    switch (command) {
      case "write-profile": {
        if (!flags.slug || !flags.ttl || !flags.body) {
          console.error("write-profile requires --slug, --ttl, and --body");
          process.exit(2);
        }
        const filePath = writeProfile({
          slug: flags.slug,
          ttl: flags.ttl,
          body: flags.body,
          config: {
            shallow_threshold: config.dedup.shallow_threshold,
            top_k: config.dedup.top_k,
          },
        });
        console.log(filePath);
        break;
      }

      case "write-playbook": {
        if (!flags.slug || !flags.body) {
          console.error("write-playbook requires --slug and --body");
          process.exit(2);
        }
        const filePath = writePlaybook({
          slug: flags.slug,
          body: flags.body,
          config: {
            shallow_threshold: config.dedup.shallow_threshold,
            top_k: config.dedup.top_k,
          },
        });
        console.log(filePath);
        break;
      }

      case "search": {
        if (!flags.query) {
          console.error("search requires --query");
          process.exit(2);
        }
        const results = search(flags.query);
        console.log(JSON.stringify({ results }, null, 2));
        break;
      }

      default:
        console.error(`Unknown command: ${command}`);
        usage();
    }
  } catch (err: any) {
    console.error(`[reflexio] error: ${err.message}`);
    process.exit(1);
  }
}

main();
