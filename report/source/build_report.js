// Builds the printed DSML project report (A4) with docx-js.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  HeadingLevel, LevelFormat, WidthType, ShadingType, BorderStyle, PageNumber, Header, Footer,
  TableOfContents, TabStopType,
} = require("docx");

const FIG = path.join(__dirname, "figs");
const OUT = process.argv[2] || path.join(__dirname, "AlphaEdge_Project_Report.docx");

// ── Look ───────────────────────────────────────────────────────────────────────
const INK = "0E1726", BODY = "262C36", MUTED = "5E6676", BLUE = "2F5BD8", TINT = "EAF0FD", RULE = "C9CED6";
const TEXT_W = 9026; // A4 (11906) minus two 1-inch margins
const BODY_FONT = "Cambria", HEAD_FONT = "Calibri";

// ── Helpers ────────────────────────────────────────────────────────────────────
// Inline markup: **bold** and _italic_. An italic underscore must not touch a letter or digit on
// its outer side, so file names such as daily_fetch.py are left alone.
function runs(text, opts = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|(?<![A-Za-z0-9])_[^_\s][^_]*?_(?![A-Za-z0-9]))/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...opts }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...opts }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...opts }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...opts }));
  return out;
}
const P = (text, o = {}) => new Paragraph({ children: runs(text), alignment: AlignmentType.JUSTIFIED, ...o });
const H1 = (text, newPage = false) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: newPage, children: [new TextRun(text)] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
const H3 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(text)] });
let listInstance = 0;
const bullets = (items) => items.map((t) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: runs(t), spacing: { after: 60 } }));
function numbered(items) {
  listInstance += 1;
  const inst = listInstance;
  return items.map((t) => new Paragraph({ numbering: { reference: "numbers", level: 0, instance: inst }, children: runs(t), spacing: { after: 60 } }));
}
const formula = (text) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 100 },
  children: [new TextRun({ text, font: "Cambria Math", size: 22 })] });

let figNo = 0, tabNo = 0;
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}
function figure(file, caption, widthPx = 540) {
  figNo += 1;
  const { w, h, data } = pngSize(path.join(FIG, file));
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 40 },
      children: [new ImageRun({ type: "png", data, transformation: { width: widthPx, height: Math.round(widthPx * h / w) },
        altText: { title: `Figure ${figNo}`, description: caption, name: file } })] }),
    new Paragraph({ style: "Caption", children: runs(`Figure ${figNo}. ${caption}`) }),
  ];
}
const border = { style: BorderStyle.SINGLE, size: 4, color: RULE };
const borders = { top: border, bottom: border, left: border, right: border };
function table(headers, rows, widths, caption, opts = {}) {
  tabNo += 1;
  const total = widths.reduce((a, b) => a + b, 0);
  if (total !== TEXT_W) throw new Error(`Table "${caption}" widths sum to ${total}, expected ${TEXT_W}`);
  // Short tables stay on one page: every row but the last keeps with the next.
  const keepTogether = rows.length <= 8;
  const cell = (text, i, header, shaded, keep) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: header ? { fill: TINT, type: ShadingType.CLEAR, color: "auto" } : shaded ? { fill: "F7F7F5", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 100, right: 100 },
    children: [new Paragraph({ keepNext: keep, spacing: { after: 0, line: 240 }, alignment: opts.align && opts.align[i] === "r" ? AlignmentType.RIGHT : AlignmentType.LEFT,
      children: runs(text, { size: 19, font: BODY_FONT, bold: header ? true : undefined, color: header ? INK : BODY }) })],
  });
  return [
    new Paragraph({ style: "Caption", keepNext: true, spacing: { before: 160, after: 60 }, children: runs(`Table ${tabNo}. ${caption}`) }),
    new Table({
      width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: widths,
      rows: [
        new TableRow({ tableHeader: true, cantSplit: true, children: headers.map((h, i) => cell(h, i, true, false, true)) }),
        ...rows.map((r, ri) => new TableRow({ cantSplit: true,
          children: r.map((c, i) => cell(c, i, false, ri % 2 === 1, keepTogether && ri < rows.length - 1)) })),
      ],
    }),
    new Paragraph({ spacing: { after: 80 }, children: [] }),
  ];
}
function keyPoint(title, text) {
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  return [
    new Table({
      width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: [TEXT_W],
      rows: [new TableRow({ cantSplit: true, children: [new TableCell({
        width: { size: TEXT_W, type: WidthType.DXA },
        borders: { top: none, bottom: none, right: none, left: { style: BorderStyle.SINGLE, size: 24, color: BLUE } },
        shading: { fill: TINT, type: ShadingType.CLEAR, color: "auto" },
        margins: { top: 100, bottom: 100, left: 200, right: 200 },
        children: [
          new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: title, bold: true, color: INK, font: HEAD_FONT, size: 22 })] }),
          new Paragraph({ spacing: { after: 0 }, children: runs(text, { size: 21 }) }),
        ],
      })] })],
    }),
    new Paragraph({ spacing: { after: 80 }, children: [] }),
  ];
}
const spacer = (after = 120) => new Paragraph({ spacing: { after }, children: [] });

// ── Title page ─────────────────────────────────────────────────────────────────
const centered = (text, o = {}) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: o.after ?? 120 },
  children: [new TextRun({ text, font: o.font || HEAD_FONT, size: o.size || 24, bold: o.bold, color: o.color || BODY, italics: o.italics })] });
const MEMBERS = ["Uzair Manjre", "Pal Gandhi", "Malav Thakor", "Shlok Barot"];
// Group members as a centred, borderless two-column table so names and roll numbers line up.
function memberTable() {
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const noBorders = { top: none, bottom: none, left: none, right: none };
  const widths = [3000, 3400];
  const cell = (text, w, o) => new TableCell({ borders: noBorders, width: { size: w, type: WidthType.DXA },
    margins: { top: 30, bottom: 30, left: 120, right: 120 },
    children: [new Paragraph({ spacing: { after: 0 }, alignment: o.align,
      children: [new TextRun({ text, font: HEAD_FONT, size: o.size, bold: o.bold, color: o.color })] })] });
  return new Table({
    alignment: AlignmentType.CENTER, width: { size: 6400, type: WidthType.DXA }, columnWidths: widths,
    borders: { top: none, bottom: none, left: none, right: none, insideHorizontal: none, insideVertical: none },
    rows: MEMBERS.map((name) => new TableRow({ children: [
      cell(name, widths[0], { align: AlignmentType.RIGHT, size: 26, bold: true, color: INK }),
      cell("Roll number: [Roll number]", widths[1], { align: AlignmentType.LEFT, size: 22, color: BODY }),
    ] })),
  });
}
const titlePage = [
  spacer(900),
  centered("DATA SCIENCE AND MACHINE LEARNING · END-SEMESTER PROJECT REPORT", { size: 20, color: BLUE, bold: true, after: 480 }),
  centered("AlphaEdge", { size: 64, bold: true, color: INK, after: 160 }),
  centered("Ranking Indian Stocks with Machine Learning", { size: 34, color: INK, after: 240 }),
  centered("Collecting and cleaning six years of market data, testing eight model versions,", { size: 22, color: MUTED, font: BODY_FONT, italics: true, after: 0 }),
  centered("and building a system that picks stocks by itself every trading day", { size: 22, color: MUTED, font: BODY_FONT, italics: true, after: 1000 }),
  centered("Submitted by", { size: 20, color: MUTED, after: 100 }),
  memberTable(),
  spacer(300),
  centered("Under the guidance of", { size: 20, color: MUTED, after: 40 }),
  centered("[Guide name]", { size: 26, bold: true, color: INK, after: 360 }),
  centered("[Department name]", { size: 22, after: 40 }),
  centered("[Institute name]", { size: 22, after: 360 }),
  centered("November 2026", { size: 22, color: MUTED, after: 120 }),
  centered("Code and data: github.com/UzairManjre/stock_predict", { size: 20, color: BLUE, after: 0 }),
];

// ── Front matter ───────────────────────────────────────────────────────────────
const abstract = [
  H1("Abstract", true),
  P("This project asks whether a program that uses only public data can choose, each trading day, a group of Indian stocks that will grow more than the market over the following month. We built **AlphaEdge**, a complete system that does this for 209 companies listed on the National Stock Exchange of India (NSE), using the NIFTY 50 index as “the market”."),
  P("We collected six years of daily prices (312,064 rows, August 2020 to October 2026) and 169,343 company announcements into a PostgreSQL database, wrote scripts that repair errors in the raw data, and built a table of 70 columns per stock per day describing trend, risk, trading activity, sector and the tone of announcements (scored with the FinBERT language model). We then trained and compared a deep neural network for time series (the Temporal Fusion Transformer), gradient-boosted decision trees (LightGBM), a model that learns the order of stocks rather than their prices, and simple rules such as momentum. Every model was tested the same way: walk-forward, on 18 months of data it had never seen."),
  P("The final model averages the rankings of four models. On the 18 test months its forecast ranges held the real outcome 80–83% of the time, as intended, and its 20 top-ranked stocks grew more than the NIFTY 50 over the next month on 68% of days, rising to 74% on days a built-in confidence flag marked as high confidence. Against the stricter benchmark of the average stock, where 50% means no skill, the picks won on 65% of days (70% on high-confidence days). The system now runs by itself after every trading day, publishes its picks on a dashboard and checks its own accuracy as results arrive."),
  P("**Keywords:** stock ranking, machine learning, gradient boosting, Temporal Fusion Transformer, walk-forward testing, ensemble, NSE", { alignment: AlignmentType.LEFT }),
];
const toc = [
  // Styled like a chapter heading but outside the outline, so it does not list itself.
  new Paragraph({ pageBreakBefore: true, spacing: { before: 440, after: 200 },
    children: [new TextRun({ text: "Contents", font: HEAD_FONT, size: 34, bold: true, color: INK })] }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
];

// ── 1. Introduction ────────────────────────────────────────────────────────────
const ch1 = [
  H1("1. Introduction", true),
  H2("1.1 The problem"),
  P("Millions of people in India invest in the stock market, and the question they face most is which companies to buy. Over long periods, most professional fund managers do not beat the market index. We wanted to know whether a program using only public data (past prices and company announcements) can do measurably better than chance. We framed the question so that it can be tested:"),
  ...keyPoint("The question", "Each trading day, can the system choose 20 stocks out of 209 such that, as a group, they grow more than the NIFTY 50 index over the next 21 trading days (about one month), more often than chance would allow?"),
  P("This framing matters. We first tried to predict each stock’s exact price change and found that no model could do that better than a coin toss. Changing the question to “which stocks will do better than the others?” is what made the problem learnable (Chapter 8)."),
  H2("1.2 What we built"),
  P("AlphaEdge is a complete pipeline from raw data to a daily list of picks (Figure 1). Scripts collect prices and announcements, repair them in a PostgreSQL database, and turn them into features. Several kinds of model are trained and compared on an honest walk-forward test, and the best design runs automatically after every trading day, with a dashboard and a record of how its past picks actually performed."),
  ...figure("fig1_pipeline.png", "The AlphaEdge pipeline. Each box is a set of Python scripts in the project repository."),
  H2("1.3 Main results"),
  ...table(["Measure", "Result", "What it means"], [
    ["Forecast range accuracy", "83% (5-day), 80% (21-day)", "Real outcomes fell inside our predicted 10th–90th percentile range as often as an honest forecast should (80%)."],
    ["Top 20 beat the NIFTY 50", "68% of all days; 74% of high-confidence days", "Share of days on which our 20 picks grew more than the index over the next month. High-confidence days are 48% of days."],
    ["Top 20 beat the average stock", "65% of all days; 70% of high-confidence days", "The stricter test: 50% would mean no skill."],
    ["Direction of a single stock", "50–52%", "Calling whether one stock goes up or down is close to a coin toss, which is why we rank instead."],
  ], [2500, 2300, 4226], "Headline results of the final model on the 18 test months (late March 2025 to September 2026)."),
];

// ── 2. Key ideas ───────────────────────────────────────────────────────────────
const ch2 = [
  H1("2. Key Ideas in Plain Words"),
  P("No background in finance is needed for this report. These are the terms we use."),
  ...table(["Term", "Meaning in this report"], [
    ["NSE and NIFTY 50", "The National Stock Exchange of India, and an index that tracks the combined value of India’s 50 largest companies. We use the NIFTY 50 as “the market”."],
    ["Return", "The percentage change in a price over a period: ₹100 to ₹105 is a 5% return."],
    ["Excess return", "A stock’s return minus the NIFTY 50’s return over the same days. Positive means it beat the market."],
    ["Trading day", "A day the exchange is open; a month has about 21 and a year about 250."],
    ["Data leak", "Information a model could not have had at the time of a prediction. A leak makes a model look much better than it is."],
    ["Percentile range", "A forecast of a low (10th percentile), middle (50th) and high (90th) value. The real outcome should fall between low and high 80% of the time."],
    ["Ranking score (IC)", "The information coefficient: the rank correlation between our ordering of the 209 stocks on a day and the order in which they actually finished. 0 means no skill; for stocks, 0.03 to 0.06 is useful."],
    ["t-statistic", "How sure we can be that an average result is not luck, given how many independent test periods we have. Above about 2 is usually taken as real."],
    ["Hit rate", "The share of days on which our 20 picks, as a group, grew more than a benchmark (the NIFTY 50 or the average stock) over the next month."],
    ["Walk-forward test", "Train on the past, test on the next few months, move forward and repeat, as a real system would be used."],
    ["Backtest", "A simulation of what the picks would have earned in the past, after trading costs."],
  ], [2100, 6926], "Terms used in this report."),
  H2("2.1 Why stock prediction is hard"),
  ...numbered([
    "**The signal is tiny.** Our best signal explains only about 0.1% of the day-to-day differences between stocks; the rest is news and noise no model can foresee.",
    "**There are fewer real examples than it seems.** The feature table has over 300,000 rows, but returns over overlapping months are not independent: six years contain only about 70 non-overlapping months, and the test period only 18. We therefore report ranges, not just single numbers.",
    "**Direction is a coin toss.** Asked whether a single stock would go up or down, our models were right 50–52% of the time.",
  ]),
  P("Our aim was therefore never to predict exact prices, but to find a small, reliable edge in the _order_ of stocks and to prove that it is not the product of luck, leaks or overfitting."),
];

// ── 3. Data collection ─────────────────────────────────────────────────────────
const ch3 = [
  H1("3. Data Collection"),
  P("We study 209 large and mid-sized NSE companies across all major sectors (banking, information technology, energy, consumer goods, pharmaceuticals, automobiles, metals and others), plus the NIFTY 50 index. Table 3 lists what we collected."),
  ...table(["Data", "Source", "How we collected it", "Amount"], [
    ["Daily prices: open, high, low, close, adjusted close, volume", "Yahoo Finance (yfinance library)", "daily_fetch.py, fetch_nifty50.py; resume from the last stored date", "312,064 rows, 31 Aug 2020 to 30 Oct 2026"],
    ["Company name, sector, industry", "Yahoo Finance and NSE", "populate_company_sectors.py", "209 companies"],
    ["Announcements: exact filing time, category, summary, PDF link", "NSE public announcements API (JSON)", "fetch_filings_metadata.py; one request per company, 1.5 s apart", "169,343 announcements"],
    ["Quarterly results filings (PDF)", "NSE archives", "download_filing_pdfs.py; newest first, resumable", "2,793 PDFs, about 20 GB"],
  ], [2350, 1950, 2826, 1900], "Data collected by the project."),
  P("**Adjusted close** is the closing price corrected for dividends and splits, so that a split does not look like a crash; every return in this project uses it. NSE blocks requests that do not look like a browser, so our scraper keeps a valid session with a small cookie manager, waits between requests and records its progress, so an interrupted download resumes where it stopped."),
  H2("3.1 Storage"),
  P("All data lives in a PostgreSQL 16 database, which lets several scripts share the data safely, enforces rules such as “one price row per stock per day”, and makes the daily update a simple insert."),
  ...table(["Table", "What it holds"], [
    ["companies", "Ticker symbol, name, sector and industry of each company"],
    ["market_data_daily", "One row per stock per trading day: prices and volume"],
    ["financial_reports, sentiment_chunks", "Announcements and the FinBERT tone scores of their text"],
    ["daily_sentiment_aggregates", "Each company’s announcement tone, summarised per day"],
    ["predictions", "The daily output: forecasts, the final rank of every stock and the confidence flag"],
    ["prediction_accuracy, headline_hit_rates", "How past predictions turned out once real prices arrived"],
  ], [3300, 5726], "Main tables in the database."),
  P("Midway through the project the work moved to a new computer and the original database could not be recovered. Because the processed feature table had been saved as a file, we rebuilt the database from it, re-downloaded the prices and re-collected the announcements from NSE’s API, which turned out to give better data (exact timestamps and categories). Since then we keep dated backups and publish the database with the code."),
];

// ── 4. Cleaning ────────────────────────────────────────────────────────────────
const ch4 = [
  H1("4. Data Cleaning"),
  P("A model trained on wrong data gives wrong answers with complete confidence, so we checked the data before building any model. We found six problems, each of which would have distorted the results."),
  ...table(["Problem we found", "What it would have caused", "How we fixed it"], [
    ["Fake price rows on market holidays (the source copies the previous day)", "Days with a false 0% move", "Dates with zero volume across most stocks are deleted"],
    ["Splits, bonus issues and demergers not adjusted (4 stocks)", "Fake one-day crashes of up to 50%", "Earlier prices rescaled so the series has no jump"],
    ["A seven-month hole in the price history", "Seven months read as one giant day", "Re-downloaded; the fetcher now resumes"],
    ["Weekend announcements dropped", "Lost company events", "Moved to the next trading day"],
    ["Missing days in the NIFTY 50 data", "The market treated as flat", "Filled from the previous trading day"],
    ["A feature that knew the future (“results week” flag)", "Results far better than reality", "Rebuilt from past information only"],
  ], [3300, 2626, 3100], "Data problems found and how each was fixed."),
  H2("4.1 The price repair script"),
  P("Cleaner/clean_prices.py runs after every price download and is safe to run repeatedly, because each repair first checks that the error is still there. It deletes **bad prints** (a single row that jumps by more than 50% and reverts the next day), deletes **fake rows** (zero-volume placeholders and whole holidays the source filled in), and rescales the history before each known **corporate action** (Table 6). Finally it lists every remaining one-day move above 20% for a person to review; genuine large moves are kept."),
  ...table(["Stock", "Date", "Reason"], [
    ["CONCOR", "1 Jan 2025", "1:4 bonus issue applied on the wrong date by the data source"],
    ["TATAMOTORS", "14 Oct 2025", "Demerger of the commercial-vehicle business"],
    ["TRENT", "1 Jan 2026", "1:2 bonus issue applied on the wrong date by the data source"],
    ["VEDL", "30 Apr 2026", "Demerger"],
  ], [2200, 2000, 4826], "Corporate actions repaired in the price history."),
  H2("4.2 Preventing leaks from the future"),
  P("Leaks are the most dangerous error in this kind of project, because they make a useless model look excellent. We applied three rules:"),
  ...bullets([
    "**Announcements count from the next trading day.** Many filings carry only a date, so we cannot tell whether they arrived before or after the market closed.",
    "**Features use only the past.** Every feature on day _t_ uses data up to day _t_; targets look only forward.",
    "**A gap between training and testing** as long as the prediction horizon (Chapter 7).",
  ]),
  P("Our first deep-learning model broke the second rule: a calendar feature told it in advance which weeks would bring quarterly results. Removing that leak cost the model all of its apparent advantage. It was the most important lesson of the project."),
];

// ── 5. Features ────────────────────────────────────────────────────────────────
const ch5 = [
  H1("5. Processing the Data: Feature Engineering"),
  P("Cleaner/build_features.py turns the clean data into one row per stock per trading day with 70 columns: identifiers, input features and the values to be predicted (Table 7)."),
  ...table(["Group", "Example features", "What they tell the model"], [
    ["Trend", "Returns over 1 to 126 days; 12-month momentum; price against its 50-day average and 52-week high", "Whether the stock has been rising or falling, and for how long"],
    ["Technical indicators", "RSI (14 days), MACD, stochastic RSI, Bollinger band width", "Whether a recent move looks stretched"],
    ["Risk", "Volatility over 20 and 60 days, average true range, beta", "How much the price usually swings, and how closely it follows the market"],
    ["Trading activity", "Volume against its 30-day average, on-balance volume, illiquidity", "Unusual buying or selling interest"],
    ["Market and sector", "NIFTY 50 return and volatility, return relative to the sector", "Whether a move is about the company or the whole market"],
    ["Announcement tone", "Decayed positive and negative tone, number of recent announcements", "What the company recently said, and how it sounded"],
    ["Calendar and identity", "Day of week, month, quarter end, sector, industry", "Seasonal effects and type of business"],
  ], [1800, 3826, 3400], "Feature groups built for every stock on every trading day."),
  H2("5.1 Announcement tone with FinBERT"),
  P("FinBERT is a language model trained on financial text [6]; we used the version published as _FinancialBERT-Sentiment-Analysis_ [7]. For each text it gives the probability of a positive, negative and neutral tone. We scored 92,165 announcement summaries and summarised them per company per day. Because news matters most on its first day, the tone decays exponentially:"),
  formula("S(t) = S(t − 1) × e^(−λ),   λ = ln 2 / half-life"),
  P("The half-life is 1.5 trading days for the largest 20% of companies by traded value, 4 days for the middle 60% and 6 days for the smallest 20%, since news about large companies is absorbed faster."),
  H2("5.2 Comparing, not measuring"),
  P("A key decision was to describe each stock relative to the others on the same day. For the ranking models every stock feature becomes a **percentile rank** among all 209 stocks that day, from 0 (lowest) to 1 (highest). This removes differences in price level and market-wide swings, leaving only what makes a stock stand out."),
  H2("5.3 What we predict"),
  P("Each model’s **target** is a stock’s excess return over the next _h_ trading days, in logarithmic form, where _P_ is the stock’s adjusted price and _N_ the NIFTY 50 level:"),
  formula("target(t, h) = ln( P(t + h) / P(t) ) − ln( N(t + h) / N(t) )"),
  P("We built targets for _h_ = 5, 10, 21 and 126 days; the final system uses _h_ = 21. For the ranking models the target is again converted into a daily percentile rank. The finished table has **310,540 rows and 70 columns** (31 August 2020 to 30 October 2026), stored as a Parquet file and extended automatically every day."),
];

// ── 6. Models ──────────────────────────────────────────────────────────────────
const ch6 = [
  H1("6. The Models We Trained"),
  P("We trained models of very different kinds, from a large neural network to a one-line rule, to see what actually works on this data (Table 8)."),
  ...table(["Model", "Type", "What it outputs", "Role in the final system"], [
    ["Temporal Fusion Transformer (TFT)", "Deep neural network for time series", "Percentile range of the 5-day excess return", "Explored in depth; not used for picks"],
    ["LightGBM quantile models", "Gradient-boosted decision trees", "Percentile range at 5, 10, 21 and 126 days", "Forecast ranges; the median is a blend member"],
    ["Rank regression", "Gradient-boosted trees on ranks", "A score for ordering stocks", "Blend member"],
    ["Momentum 12-1", "Rule, no training", "12-month return, skipping the last month", "Blend member"],
    ["Per-stock history", "Simple statistic", "Each stock’s typical past 21-day excess return", "Blend member"],
    ["LambdaRank, CatBoost, tuned LightGBM", "Ranking and boosting variants", "Scores for ordering stocks", "Tested; did not pass our checks"],
  ], [2150, 2050, 2626, 2200], "Overview of the models."),
  H2("6.1 Temporal Fusion Transformer"),
  P("The TFT [1] is a neural network designed for forecasting time series. For each stock it reads the last 60 trading days: a _variable selection network_ learns which inputs matter, an LSTM [13] summarises the sequence, and an _attention_ layer [14] picks out the most relevant past days. It outputs three percentiles, trained with the quantile loss of Section 6.2, and one model serves all 209 stocks."),
  ...table(["Setting", "Value", "Setting", "Value"], [
    ["Look-back window", "60 trading days", "Hidden size / attention heads", "128 / 4"],
    ["LSTM layers / dropout", "2 / 0.25", "Optimiser, learning rate", "Adam, 0.001"],
    ["Batch size", "256", "Training / validation data", "Sep 2020–Jun 2024 / Jul 2024–Jun 2025"],
  ], [2300, 2213, 2300, 2213], "Settings of the final TFT version (PyTorch Forecasting, NVIDIA RTX 3080 GPU)."),
  P("The first version beat a simple baseline by 0.75%, but only through the leaked “results week” feature; with the leak and data errors fixed it was 0.37% _worse_ than the baseline. Investigating why, we found that _weight decay_, a penalty that pulls weights toward zero, had shrunk the variable-selection layer to about 10⁻⁴¹, switching every input off. Without weight decay the model came back to life but memorised: training error fell 8.8% while error on new data rose 4.5% (Figure 2). Daily stock data simply holds too little repeatable pattern for a model this large."),
  ...figure("fig3_tft_loss.png", "TFT training without weight decay: the model improves on its training data (orange) while getting worse on new data (blue), a clear sign of overfitting.", 500),
  H2("6.2 Gradient-boosted trees (LightGBM)"),
  P("**Gradient boosting** builds hundreds of small decision trees one after another, each trained to correct the errors of the trees before it [2]. It suits tables of features like ours and trains in minutes. We trained one LightGBM model per percentile with the **quantile (pinball) loss** [12]:"),
  formula("loss = q × (y − ŷ)  if  y ≥ ŷ,      otherwise  (1 − q) × (ŷ − y)"),
  P("Here _y_ is the outcome, _ŷ_ the forecast and _q_ the percentile. For _q_ = 0.1, forecasting too high costs nine times more than forecasting too low, which pushes the forecast down until about 10% of outcomes fall below it. The models use 37 inputs (29 numeric features and 8 categories), a learning rate of 0.05, 63 leaves per tree, 80% of features and rows sampled per tree, and **early stopping**: training stops once error on the last 42 trading days before the test has not improved for 50 rounds."),
  H2("6.3 Rank regression"),
  P("This LightGBM model learns the _order_ of stocks. Its target is the daily percentile rank of each stock’s future excess return, and its inputs are 33 stock features, each converted into a daily rank, plus the sector. The stock’s name, the month and raw price levels are deliberately left out so it cannot memorise which stocks did well before. Its settings are cautious: learning rate 0.03, 31 leaves, at least 200 rows per leaf and an L2 penalty of 1.0."),
  H2("6.4 Momentum and per-stock history"),
  P("**Momentum** is a rule studied in finance for over thirty years [5]: stocks that rose most over the past year tend to keep doing slightly better for a while. We use the return over the last 12 months, skipping the latest month, which tends to reverse briefly:"),
  formula("momentum(t) = ln( P(t − 21) / P(t − 252) )"),
  P("It needs no training and so cannot overfit. **Per-stock history** is each stock’s median 21-day excess return over its training data. It is weak alone, but its mistakes differ from the other members’, which makes it useful in a blend."),
  H2("6.5 The final model: a blend of four"),
  ...numbered([
    "Momentum, rank regression, the LightGBM median forecast and per-stock history each score all 209 stocks; each score becomes a percentile rank within the day (a missing score is set to 0.5).",
    "The four ranks are averaged with equal weights.",
    "The average is smoothed over each stock’s last 5 trading days, which reduces noise and how often the picks change.",
    "The 20 stocks with the highest smoothed score are the picks.",
  ]),
  P("Averaging works because the members make different mistakes: their daily ranking scores are only weakly correlated (0.04 to 0.54). Weights learned from past data did worse than a plain average, because the weights themselves overfit."),
  H2("6.6 The confidence flag"),
  P("Each day we measure how much the four members agree, as the average Spearman rank correlation between every pair of their rankings. A day is **high confidence** when agreement is above its median over the previous 126 trading days (about six months; at least 21 days needed). We fixed this rule before measuring its effect, so it could not be tuned to the test."),
  H2("6.7 What did not make it"),
  ...bullets([
    "**LambdaRank** [10], a learning-to-rank method from web search, ranked stocks slightly backwards on the test (ranking score −0.027).",
    "**Automatic tuning with Optuna** [4] tried 100 settings for a LightGBM ranker. The winner scored 0.059 on its tuning data (2023 to January 2025) but 0.009 on the test: it had tuned itself to the past.",
    "**A stricter second tuning run**, including a **CatBoost** model [3], fixed a pass-or-fail rule in advance. Both candidates failed it, so neither was used and the test was never used to choose between them.",
  ]),
];

// ── 7. Training and testing ────────────────────────────────────────────────────
const ch7 = [
  H1("7. How We Train and Test"),
  H2("7.1 Walk-forward testing"),
  P("The test period is split into six rounds of about three months (63 trading days). In every round each model is trained from scratch on all data before the round and then predicts every day of the round; the rounds move forward in time (Figure 3). Together they cover **378 trading days, from late March 2025 to September 2026**, and every model receives exactly the same test."),
  ...figure("fig2_walkforward.png", "Walk-forward testing: train on all earlier data, leave a 21-day gap so no answer leaks in, test on the next three months.", 520),
  P("Two details keep the test honest. A 21-day target on the last training day looks into the test period, so we leave a **gap (embargo)** as long as the horizon between training and test data [8]. And the data used for early stopping comes from the end of the training block, so the test data is never used for any decision."),
  H2("7.2 How we measure success"),
  ...table(["Measure", "How it is computed", "Good result"], [
    ["Ranking score (IC)", "Spearman rank correlation between our scores and the real excess returns of all stocks, each day, then averaged", "0.03–0.06"],
    ["t-statistic", "Mean IC divided by its standard error, using only non-overlapping periods (one day in 5, or one in 21)", "Above about 2"],
    ["Hit rate", "Share of days on which the top 20, as an equal-weight group, beat the benchmark over the next 21 days", "Above the benchmark’s own rate"],
    ["Range accuracy", "Share of outcomes inside the forecast 10th–90th percentile range", "Close to 80%"],
    ["Backtest return", "Buy the top 20 equally, re-buy each week or month, subtract 0.18% on every trade (brokerage, tax and slippage)", "Above the NIFTY 50"],
  ], [1900, 5326, 1800], "How results are measured."),
  H2("7.3 Rules that keep us honest"),
  ...bullets([
    "**One test, used once.** We did not keep adjusting models until they looked good on the test months; that would turn the test into training data.",
    "**Rules fixed in advance** for the confidence flag and for accepting tuned models.",
    "**Uncertainty reported.** Hit rates come with 90% ranges from a moving-block bootstrap [11], which resamples test days in blocks of 21 so related neighbouring days stay together.",
  ]),
  P("**Tools.** Python 3.10 (66 scripts), pandas, PostgreSQL 16, LightGBM, CatBoost, PyTorch Forecasting, Optuna, Hugging Face Transformers, Streamlit and git. Neural networks were trained on an NVIDIA RTX 3080 GPU; the tree models train in minutes on the processor."),
];

// ── 8. Journey ─────────────────────────────────────────────────────────────────
const ch8 = [
  H1("8. The Development Journey: Eight Iterations"),
  P("The final design came out of eight iterations, each answering a question raised by the one before. Three were dead ends; we keep them because each taught us something we used later."),
  ...table(["#", "What we did", "What we found", "Outcome"], [
    ["1", "Trained a TFT to forecast 5-day returns", "Beat a baseline by 0.75%, but only through a leaked feature", "Dead end"],
    ["2", "Fixed the leak and the six data problems", "Honest result: no better than a guess", "Kept the fixes"],
    ["3", "Asked “which stocks will do best?” instead of “how much will each move?”", "Rankings matched reality measurably (ranking score up to 0.06)", "Kept"],
    ["4", "Tested momentum, a one-line rule", "Beat every trained model: +27.4% in the backtest vs −3.8% for the NIFTY 50", "Kept"],
    ["5", "Averaged four different models", "5-day reliability rose from t = 1.4–2.2 for single models to t = 3.0", "Kept"],
    ["6", "Diagnosed why the neural network learned nothing", "Weight decay had switched off its inputs; without it, it memorised", "Dead end"],
    ["7", "Automatic tuning with Optuna (twice)", "Overfitted (0.059 → 0.009); the stricter run rejected its candidates", "Dead end"],
    ["8", "Added the confidence flag and the live system", "74% of high-confidence days beat the NIFTY 50; checked daily", "Final"],
  ], [450, 3300, 3876, 1400], "The eight iterations of the project."),
  P("**The turning point (iteration 3).** An investor does not need to know that a stock will rise by exactly 2.3%, only which stocks are likely to do better than the others. Most of a stock’s movement comes from the market as a whole, which every stock shares; predicting the _order_ removes it and leaves the smaller, more stable differences between companies. The ranking score was clearly positive where direction accuracy had been stuck at 50%."),
  P("**Simple beats complex, and many beat one (iterations 4 and 5).** On noisy data a rule with no parameters can beat a complex model because it cannot overfit, and combining models that make different mistakes is more reliable than picking the single best one (Figure 4)."),
  ...figure("fig4_tstat.png", "Reliability of the daily ranking at the 5-day horizon over the 18 test months (t-statistic of the ranking score). The plain average of four models is far more reliable than any one of them.", 500),
];

// ── 9. Results ─────────────────────────────────────────────────────────────────
const ch9 = [
  H1("9. Results"),
  P("All results come from the 18 test months and are reproduced by scripts in the repository (saved under results/)."),
  H2("9.1 Forecast ranges and ranking quality"),
  P("The real outcome fell inside the LightGBM 10th–90th percentile range **83% of the time at 5 days and 80% at 21 days**, close to the 80% an honest forecast should achieve. The middle forecast, however, called a single stock’s direction correctly only 50–52% of the time, which confirms that the useful information lies in ranges and rankings, not point forecasts. Table 12 compares the ranking models at 21 days."),
  ...table(["Model", "Ranking score (IC)", "t-statistic", "Beat the average stock", "Beat the NIFTY 50"], [
    ["Momentum 12-1", "0.052", "2.22", "59.9%", "66.3%"],
    ["LightGBM median forecast", "0.041", "0.74", "58.1%", "67.4%"],
    ["Per-stock history", "0.020", "0.86", "44.6%", "54.9%"],
    ["Rank regression", "0.017", "0.69", "41.9%", "53.1%"],
    ["LambdaRank", "−0.027", "−1.42", "41.9%", "49.1%"],
    ["**Final blend (smoothed)**", "**0.057**", "**1.85**", "**65.0%**", "**68.4%**"],
  ], [2826, 1500, 1300, 1700, 1700], "Ranking results at the 21-day horizon over 378 test days. Hit rates: how often the top 20 beat each benchmark over the next 21 trading days.", { align: ["l", "r", "r", "r", "r"] }),
  P("The final blend has the highest ranking score and hit rates of any model. At 21 days the test contains only 18 non-overlapping periods, so t-statistics are lower than at 5 days even for good signals; the blend’s 1.85 should be read with its 5-day value of 3.02."),
  H2("9.2 Hit rates and the confidence flag"),
  P("In this period most stocks happened to beat the NIFTY 50: the average of our 209 stocks did so on 63% of days. Figure 5 therefore shows both benchmarks: the NIFTY 50 (Panel A) and the stricter average stock (Panel B), where 50% means no skill."),
  ...figure("fig6_hitrate.png", "How often the final model’s 20 picks grew more than the NIFTY 50 (A) and more than the average of all 209 stocks (B) over the next 21 trading days.", 500),
  ...table(["Days", "Share of days", "Beat the NIFTY 50", "Beat the average stock", "Average excess return"], [
    ["High-confidence", "48% (172 days)", "74% (64–90%)", "70% (60–84%)", "+2.1% per month"],
    ["Other", "52% (184 days)", "68% (48–83%)", "66% (46–82%)", "+0.8% per month"],
    ["All", "100% (356 days)", "71% (59–83%)", "68% (56–80%)", "+1.4% per month"],
  ], [1700, 1800, 1900, 1926, 1700], "The confidence flag on the test months, with 90% block-bootstrap ranges. The first test month is excluded because the flag needs 21 days of history.", { align: ["l", "r", "r", "r", "r"] }),
  P("On high-confidence days the picks did better on every measure, with more than twice the average excess return. These days add up to only about eight independent months, so the true rate could lie anywhere from 64% to 90%: the flag is a guide to which days deserve more trust, not a guarantee."),
  H2("9.3 Backtest"),
  P("Figure 6 follows ₹100 invested at the end of March 2025 and re-invested every week in each method’s top 20, after costs. Momentum grew it to ₹127 while the NIFTY 50 fell to ₹96. Buying all 209 stocks equally gave ₹113, so part of any strategy’s gain in this period came simply from holding mid-sized companies. Re-bought every 21 trading days, the final blend gained **33.7%**, against 25.3% for all stocks bought equally and 5.0% for the NIFTY 50."),
  ...figure("fig5_growth.png", "Backtest of ₹100 from the end of March 2025 to October 2026, re-bought every week, after a 0.18% cost on every trade.", 500),
  H2("9.4 What announcements tell us"),
  P("Across 127,315 announcement events, quarterly results roughly double the size of the next two days’ price move (3.8% against 1.9% on an ordinary day), but prices go up and down about equally often afterwards. FinBERT tone helps only on the day itself (+0.2% after positive summaries, −0.5% after negative ones). Announcements tell us _when_ a stock will move, not _which way_, which is why they play only a small part in the final model."),
];

// ── 10. System ─────────────────────────────────────────────────────────────────
const ch10 = [
  H1("10. How the System Works Day to Day"),
  P("The final model is part of a system that runs by itself after every trading day:"),
  ...numbered([
    "**18:30 on weekdays**, Windows Task Scheduler starts run_live_tracking.py, which starts PostgreSQL if needed.",
    "**Fetch and repair:** daily_fetch.py downloads the day’s prices and clean_prices.py repairs them.",
    "**Features:** build_features.py adds the new day to the feature table.",
    "**Retrain every 15 days:** when the models are 15 or more days old, they are retrained on all data to date.",
    "**Predict:** predict.py ranks all 209 stocks with the final blend, sets the confidence flag and saves the results.",
    "**Drift check:** a Kolmogorov–Smirnov test warns if today’s features differ markedly from the training data.",
    "**Score the past:** accuracy_checker.py compares every prediction that has reached 5 or 21 trading days with what really happened.",
  ]),
  P("Each run writes a dated log and takes about two minutes. A Streamlit dashboard (dashboard/app.py) reads the database and shows the results on seven pages (Table 14)."),
  ...table(["Page", "What it shows"], [
    ["Overview", "Market summary and the state of the latest run"],
    ["Top Picks", "Today’s ranking by the final model, with a banner saying whether today is a high-confidence day"],
    ["Stock Deep Dive", "Price history, features and forecast range for any stock"],
    ["Sector Analysis", "How each sector is ranked and how it has performed"],
    ["News and Sentiment", "Recent announcements and their FinBERT tone"],
    ["Accuracy", "Live tracking of how often past top-20 picks beat the NIFTY 50, beside the test-period values"],
    ["Pipeline Status", "When each stage last ran and whether it succeeded"],
  ], [2400, 6626], "Pages of the AlphaEdge dashboard."),
  P("Because the system scores its own predictions, testing continues beyond the 18 historical months. Live tracking started on 17 October 2026; the first 21-day results arrive in mid-November 2026, after which the Accuracy page shows live hit rates next to the research values."),
  ...table(["Folder or file", "Contents"], [
    ["Scaper/", "Data collection: NSE announcements, filing metadata, PDF downloads, cookie handling"],
    ["Cleaner/", "Cleaning, features, training (train_tft.py, train_lgbm.py, final_model.py), evaluation (evaluate.py, ensemble_study.py, confidence_study.py), prediction and accuracy checks"],
    ["dashboard/", "The Streamlit dashboard"],
    ["artifacts/, results/", "Trained models and the agreement history; every result table and chart in this report"],
    ["run_daily_update.py", "The daily orchestrator (run_live_tracking.py wraps it for the scheduler)"],
  ], [2400, 6626], "Layout of the project repository."),
  P("The code is public at **github.com/UzairManjre/stock_predict**, with the database and large data files attached as downloadable releases so every result can be reproduced."),
];

// ── 11–12. Limitations, conclusion ─────────────────────────────────────────────
const ch11 = [
  H1("11. Limitations and Future Work"),
  ...bullets([
    "**Survivorship bias.** We study companies that exist today; firms that collapsed or were delisted since 2020 are missing, which flatters any stock-picking result.",
    "**A short test.** Eighteen months is short in markets: the 74% could really be anywhere from 64% to 90%, and the system has not been tested through a long crash.",
    "**A good period for mid-sized companies.** Most stocks beat the NIFTY 50 during the test, which is why we also report results against the average stock.",
    "**Patterns change.** From August 2024 to January 2025 every model ranked stocks backwards.",
    "**Not investment advice.** Real trading would also need taxes, position sizes and risk limits.",
  ]),
  P("**Future work.** Read the 2,793 downloaded results PDFs in full with FinBERT; add delisted companies and more years of history; use the live results from November 2026 as a second, truly unseen test; and study whether the confidence flag can also decide how much to invest."),
];
const ch12 = [
  H1("12. Conclusion"),
  P("We built a complete system that collects six years of Indian market data, repairs it, turns it into 70 columns per stock per day, trains and compares several kinds of model under one honest walk-forward test, and runs by itself every trading day. The main findings are simple: predicting a stock’s exact price change is close to impossible, but ranking stocks against each other is not; a plain average of four different models ranks more reliably than any single model, including a large neural network; and the models’ agreement tells us which days to trust. On 18 months of unseen data the final model’s 20 picks beat the NIFTY 50 over the next month on 68% of days and on 74% of high-confidence days, and beat the average stock on 65% and 70% respectively."),
  P("Just as valuable were the habits the project taught us: clean the data first, hunt for leaks, test once on unseen data, report uncertainty and keep the failures. The system is now running live, and from November 2026 it will show, day by day, whether these results hold up."),
];

// ── References and appendix ────────────────────────────────────────────────────
const refs = [
  "B. Lim, S. Ö. Arık, N. Loeff and T. Pfister, “Temporal Fusion Transformers for interpretable multi-horizon time series forecasting,” International Journal of Forecasting, vol. 37, no. 4, pp. 1748–1764, 2021.",
  "G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye and T.-Y. Liu, “LightGBM: A highly efficient gradient boosting decision tree,” in Advances in Neural Information Processing Systems 30, 2017.",
  "L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush and A. Gulin, “CatBoost: Unbiased boosting with categorical features,” in Advances in Neural Information Processing Systems 31, 2018.",
  "T. Akiba, S. Sano, T. Yanase, T. Ohta and M. Koyama, “Optuna: A next-generation hyperparameter optimization framework,” in Proc. 25th ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining, 2019.",
  "N. Jegadeesh and S. Titman, “Returns to buying winners and selling losers: Implications for stock market efficiency,” The Journal of Finance, vol. 48, no. 1, pp. 65–91, 1993.",
  "D. Araci, “FinBERT: Financial sentiment analysis with pre-trained language models,” arXiv:1908.10063, 2019.",
  "A. Hazourli, “FinancialBERT-Sentiment-Analysis,” model card, Hugging Face (ahmedrachid/FinancialBERT-Sentiment-Analysis), 2022.",
  "M. López de Prado, Advances in Financial Machine Learning. Hoboken, NJ: Wiley, 2018.",
  "R. C. Grinold and R. N. Kahn, Active Portfolio Management, 2nd ed. New York: McGraw-Hill, 2000.",
  "C. J. C. Burges, “From RankNet to LambdaRank to LambdaMART: An overview,” Microsoft Research, Tech. Rep. MSR-TR-2010-82, 2010.",
  "H. R. Künsch, “The jackknife and the bootstrap for general stationary observations,” The Annals of Statistics, vol. 17, no. 3, pp. 1217–1241, 1989.",
  "R. Koenker and G. Bassett, “Regression quantiles,” Econometrica, vol. 46, no. 1, pp. 33–50, 1978.",
  "S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.",
  "A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser and I. Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems 30, 2017.",
];
const references = [
  H1("References"),
  ...refs.map((r, i) => new Paragraph({ spacing: { after: 80 }, indent: { left: 540, hanging: 540 },
    tabStops: [{ type: TabStopType.LEFT, position: 540 }],
    children: [new TextRun({ text: `[${i + 1}]\t`, size: 20 }), new TextRun({ text: r, size: 20 })] })),
];
const code = (text) => new Paragraph({ spacing: { after: 30 }, indent: { left: 360 },
  children: [new TextRun({ text, font: "Consolas", size: 17, color: INK })] });
const appendix = [
  H1("Appendix A. Running the Project"),
  P("On Windows with Python 3.10 and PostgreSQL 16, from the project folder:", { alignment: AlignmentType.LEFT }),
  H3("Get the code and data"),
  code("git clone https://github.com/UzairManjre/stock_predict.git"),
  code("python -m venv venv  &&  venv\\Scripts\\pip install -r requirements.txt"),
  P("Download both releases from the repository’s Releases page, unzip the data archive into the project folder, and restore the database:", { alignment: AlignmentType.LEFT }),
  code("createdb -h localhost -p 5433 -U postgres stock_predict"),
  code("pg_restore -h localhost -p 5433 -U postgres -d stock_predict --no-owner stock_predict_2026-11-01.dump"),
  H3("Reproduce the results"),
  P("The walk-forward test of every model, the blends and hit rates, and the confidence flag:", { alignment: AlignmentType.LEFT }),
  code("venv\\Scripts\\python Cleaner\\evaluate.py"),
  code("venv\\Scripts\\python Cleaner\\ensemble_study.py --horizon 21"),
  code("venv\\Scripts\\python Cleaner\\confidence_study.py"),
  H3("Run the live system"),
  P("One full daily run, the 18:30 weekday schedule, and the dashboard:", { alignment: AlignmentType.LEFT }),
  code("venv\\Scripts\\python run_daily_update.py"),
  code("powershell -File setup_live_tracking.ps1"),
  code("venv\\Scripts\\streamlit run dashboard\\app.py"),
];

// ── Document ───────────────────────────────────────────────────────────────────
const headerPara = new Paragraph({ alignment: AlignmentType.RIGHT, border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE, space: 4 } },
  children: [new TextRun({ text: "AlphaEdge: Ranking Indian Stocks with Machine Learning", font: HEAD_FONT, size: 17, color: MUTED })] });
const footerPara = new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], font: HEAD_FONT, size: 18, color: MUTED })] });

const doc = new Document({
  creator: "Uzair Manjre, Pal Gandhi, Malav Thakor, Shlok Barot",
  title: "AlphaEdge: Ranking Indian Stocks with Machine Learning",
  description: "DSML end-semester project report",
  styles: {
    default: { document: { run: { font: BODY_FONT, size: 22, color: BODY }, paragraph: { spacing: { after: 120, line: 276 } } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: HEAD_FONT, size: 34, bold: true, color: INK },
        paragraph: { spacing: { before: 440, after: 200 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: HEAD_FONT, size: 26, bold: true, color: BLUE },
        paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 1, keepNext: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: HEAD_FONT, size: 22, bold: true, color: INK },
        paragraph: { spacing: { before: 160, after: 60 }, outlineLevel: 2, keepNext: true } },
      { id: "Caption", name: "Caption", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: HEAD_FONT, size: 19, italics: true, color: MUTED },
        paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 40, after: 160 } } },
      { id: "TOC1", name: "toc 1", basedOn: "Normal", next: "Normal",
        run: { bold: true, color: INK }, paragraph: { spacing: { before: 60, after: 0, line: 252 } } },
      { id: "TOC2", name: "toc 2", basedOn: "Normal", next: "Normal",
        paragraph: { spacing: { before: 0, after: 0, line: 252 }, indent: { left: 360 } } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440, header: 708, footer: 708 } },
      titlePage: true,
    },
    headers: { default: new Header({ children: [headerPara] }), first: new Header({ children: [new Paragraph({ children: [] })] }) },
    footers: { default: new Footer({ children: [footerPara] }), first: new Footer({ children: [new Paragraph({ children: [] })] }) },
    children: [
      ...titlePage, ...abstract, ...toc,
      ...ch1, ...ch2, ...ch3, ...ch4, ...ch5, ...ch6, ...ch7, ...ch8, ...ch9, ...ch10, ...ch11, ...ch12,
      ...references, ...appendix,
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log(`wrote ${OUT} (${(buf.length / 1024).toFixed(0)} KB, ${figNo} figures, ${tabNo} tables)`);
});
