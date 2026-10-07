# Putting the site online

The site in `web/` is static: HTML, CSS, and JavaScript files. No server, no
build step, no npm, no database. That means it will run on almost anything,
including several free hosts, and it means there is very little that can break
later.

If you only read one section, read the first one.

---

## Option 1: GitHub Pages (free, already set up)

This is the one to use. The repository already contains the workflow; you just
need to switch Pages on.

1. Go to **Settings → Pages** in this repository on GitHub.
2. Under **Build and deployment → Source**, choose **GitHub Actions**.
3. Go to **Actions → Deploy site → Run workflow** to publish immediately.
   After this, every push to `main` republishes automatically.

> **Until step 2 is done, the Deploy site workflow fails** — on the
> `Run actions/configure-pages` step, with a message about not being able to
> find the Pages site. That is expected rather than a problem with the
> workflow: the action asks GitHub where to deploy, and nowhere has been set
> up yet. The step before it, which checks the data bundle, passes.

Your URL will be:

```
https://melvthegoat.github.io/Uplift-Modelling-Decision/
```

**The capitals matter.** GitHub repository names are case-insensitive when you
browse to them, so `github.com/melvthegoat/uplift-modelling-decision` works
fine — but the Pages path is case-*sensitive* and the all-lowercase version
returns a flat 404 with no hint as to why. Copy the URL exactly as above.

To publish without waiting for a push, go to **Actions → Deploy site → Run
workflow**.

### What the workflow does

`.github/workflows/pages.yml` runs two jobs. The second uploads `web/` to
Pages. The first is a safety check worth understanding, because it will
eventually stop you shipping something wrong.

`web/data.json` is a *build artefact that is committed to the repository*. It
is generated from `results/` by `scripts/build_web_data.py`. That is convenient
— a fresh clone can serve the site immediately — but it creates a trap: re-run
the study, commit new results, forget to rebuild the bundle, and the site goes
on showing the old numbers with no error anywhere.

So the workflow rebuilds the bundle from `results/` and refuses to deploy if it
differs from the committed one. If the build fails with *"web/data.json is
stale"*, the fix is:

```bash
python scripts/build_web_data.py
git add web/data.json
git commit -m "Refresh the site's data bundle"
```

---

## Option 2: Netlify, Vercel or Cloudflare Pages (free)

All three work with no configuration beyond pointing them at the right folder.
Use one of these if you want a custom domain or preview deployments on pull
requests.

Connect the repository, then set:

| Setting | Value |
|---|---|
| Build command | *leave empty* |
| Publish / output directory | `web` |

There is nothing to build, so an empty build command is correct rather than a
shortcut. If the host insists on something, `echo nothing to build` is a valid
answer.

The one thing to remember: these hosts do **not** run the staleness check from
option 1. If you re-run the study, rebuild `web/data.json` yourself before
pushing.

---

## Option 3: Any web server, or a file you hand someone

The `web/` directory is self-contained. Copy it to an S3 bucket, an nginx
document root, a shared drive, or a USB stick — anywhere that can serve files
over HTTP will serve this.

```bash
# Locally, to check it before publishing:
python -m http.server -d web 8000
# then open http://localhost:8000
```

**You cannot just double-click `index.html`.** The page uses JavaScript modules
and `fetch`, and browsers refuse to run either from a `file://` path for
security reasons. You will get a blank page and a CORS error in the console.
The site detects this case and says so, but the fix is always "serve it over
HTTP", which is what the command above does.

---

## Updating the site after re-running the study

```bash
python -m src.cli all               # regenerate results/
python scripts/build_web_data.py    # regenerate web/data.json
python -m http.server -d web 8000   # check it locally
git add results web/data.json
git commit -m "Refresh results and the site bundle"
git push
```

The push triggers a redeploy. Give it a minute or two.

---

## How big is it, and will it cost anything?

No. The whole site is well under a megabyte:

| | |
|---|---|
| `web/data.json` | 116 KB — every number in the study |
| `web/sample-campaign.csv` | 504 KB — the demo file, fetched only if asked for |
| JavaScript | ~190 KB, uncompressed, no dependencies |
| CSS | 18 KB |
| HTML | 3 KB |

Under a megabyte all in, and the sample file is the largest piece — it is
fetched on demand, so a visitor who never clicks "load a sample" never
downloads it.

All free tiers listed above cover this comfortably. There is no server process,
no database, and nothing that runs when nobody is looking at the page, so there
is nothing to meter.

---

## Things that will go wrong

**The page is blank and the console says "Failed to fetch dynamically imported
module".** You are opening the file directly instead of serving it. See option
3.

**The page says "The findings could not be loaded".** `web/data.json` is
missing or not being served. Check it exists, and that your host is publishing
the `web` directory rather than the repository root.

**The site deploys but shows old numbers.** The data bundle is stale. Rebuild
it with `python scripts/build_web_data.py` and commit.

**The Deploy site action fails on `Run actions/configure-pages`.** Pages is not
enabled, or its source is not set to "GitHub Actions". See option 1, step 2.
This is the expected state of a fresh clone.

**Everything works locally but 404s on GitHub Pages.** The site is served from
a subdirectory (`/uplift-modelling-decision/`) rather than the domain root.
Every link and asset path in the site is relative, so this is handled — but if
you have added an absolute path starting with `/`, that is what broke it.
