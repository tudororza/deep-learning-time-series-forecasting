#!/usr/bin/env python3
"""Render the final report PDF from the reproducible experiment results."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2F6B9A")
PALE = colors.HexColor("#EAF2F8")
GREEN = colors.HexColor("#287A5A")
LIGHT_GREEN = colors.HexColor("#E8F5EE")
GRAY = colors.HexColor("#5A6772")
LIGHT_GRAY = colors.HexColor("#F2F4F6")


class ArchitectureDiagram(Flowable):
    def __init__(self, width: float, height: float = 82 * mm) -> None:
        super().__init__()
        self.width = width
        self.height = height

    def draw(self) -> None:
        canvas = self.canv
        boxes = [
            (4, 48, 31, 20, "168-hour history\ntarget + covariates"),
            (41, 48, 25, 20, "LSTM\n64 units"),
            (72, 48, 27, 20, "LayerNorm\nfinal state"),
            (106, 48, 30, 20, "Residual\ndecoder"),
            (143, 48, 27, 20, "Add seasonal\nreference"),
            (177, 48, 25, 20, "24-hour\nforecast"),
            (74, 10, 58, 20, "Known future covariates\n+ learned series embedding"),
        ]
        scale = self.width / (206 * mm)
        canvas.saveState()
        canvas.scale(scale, scale)
        canvas.setLineWidth(1.2)
        for x, y, w, h, label in boxes:
            canvas.setFillColor(PALE if y > 20 else LIGHT_GREEN)
            canvas.setStrokeColor(BLUE if y > 20 else GREEN)
            canvas.roundRect(x * mm, y * mm, w * mm, h * mm, 3 * mm, fill=1)
            canvas.setFillColor(NAVY)
            canvas.setFont("Helvetica-Bold", 8)
            lines = label.split("\n")
            for line_no, line in enumerate(lines):
                text_x = (x + w / 2) * mm - stringWidth(line, "Helvetica-Bold", 8) / 2
                canvas.drawString(text_x, (y + h / 2 + 2 - line_no * 4) * mm, line)
        arrows = [((35, 58), (41, 58)), ((66, 58), (72, 58)), ((99, 58), (106, 58)),
                  ((136, 58), (143, 58)), ((170, 58), (177, 58)), ((103, 30), (118, 48))]
        canvas.setStrokeColor(GRAY)
        canvas.setFillColor(GRAY)
        for (x1, y1), (x2, y2) in arrows:
            canvas.line(x1 * mm, y1 * mm, x2 * mm, y2 * mm)
            canvas.circle(x2 * mm, y2 * mm, 1.2 * mm, fill=1, stroke=0)
        canvas.restoreState()


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=24, leading=28, textColor=NAVY, alignment=TA_LEFT, spaceAfter=5 * mm
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontName="Helvetica",
            fontSize=11, leading=15, textColor=GRAY, spaceAfter=9 * mm
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=16, leading=19, textColor=NAVY, spaceBefore=3 * mm, spaceAfter=3 * mm
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=11.5, leading=14, textColor=BLUE, spaceBefore=3 * mm, spaceAfter=1.5 * mm
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica",
            fontSize=9.4, leading=13.3, textColor=colors.HexColor("#202830"),
            alignment=TA_LEFT, spaceAfter=2.5 * mm
        ),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8.2, leading=11, textColor=GRAY, spaceAfter=2 * mm
        ),
        "abstract": ParagraphStyle(
            "Abstract", parent=base["BodyText"], fontName="Helvetica",
            fontSize=10, leading=14.5, textColor=NAVY, borderColor=BLUE,
            borderWidth=1, borderPadding=10, backColor=PALE, spaceAfter=7 * mm
        ),
        "callout": ParagraphStyle(
            "Callout", parent=base["BodyText"], fontName="Helvetica-Bold",
            fontSize=12, leading=16, textColor=GREEN, borderColor=GREEN,
            borderWidth=1, borderPadding=9, backColor=LIGHT_GREEN, spaceAfter=6 * mm
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["BodyText"], fontName="Helvetica-Oblique",
            fontSize=8, leading=10, textColor=GRAY, alignment=TA_CENTER, spaceAfter=3 * mm
        ),
        "reference": ParagraphStyle(
            "Reference", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8.5, leading=12, leftIndent=5 * mm, firstLineIndent=-5 * mm,
            textColor=colors.HexColor("#303840"), spaceAfter=3 * mm
        ),
    }


def results_table(data: list[list[str]], widths: list[float]) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C9D1D8")),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def page_decor(canvas, document) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#D6DDE3"))
    canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
    canvas.setFillColor(GRAY)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(18 * mm, 9.5 * mm, "TU Darmstadt - Deep Learning SS26")
    page = str(document.page)
    canvas.drawRightString(width - 18 * mm, 9.5 * mm, page)
    canvas.restoreState()


def build(output: Path) -> None:
    s = styles()
    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=19 * mm,
        title="Residual LSTM Forecasting for Multivariate Operational Time Series",
        author="Tudor Orza, Marie Reinhold, Felix Besser",
    )
    story: list[Flowable] = []

    story += [
        Spacer(1, 9 * mm),
        Paragraph("Residual LSTM Forecasting for<br/>Multivariate Operational Time Series", s["title"]),
        Paragraph(
            "Tudor Orza &nbsp;&nbsp; Marie Reinhold &nbsp;&nbsp; Felix Besser<br/>"
            "Deep Learning Bonus Project - Summer Semester 2026",
            s["subtitle"],
        ),
        Paragraph(
            "<b>Abstract.</b> We forecast hourly operational load for 96 anonymized units. "
            "The required system predicts 336 hours recursively in 24-hour blocks and is "
            "evaluated with WAPE. Our compact PyTorch LSTM predicts corrections to a strong "
            "four-week seasonal mean, using known future covariates and learned unit "
            "embeddings. On a chronological holdout it reaches WAPE <b>0.2413</b>, compared "
            "with 0.3418 for the seasonal mean and 0.5450 for last-value persistence. We "
            "also transfer the architecture to joint Jena temperature and humidity forecasting.",
            s["abstract"],
        ),
        Paragraph("0.2413 internal WAPE - 29% below the strongest seasonal baseline", s["callout"]),
        Paragraph("1. Introduction", s["h1"]),
        Paragraph(
            "Operational load forecasting supports staffing, capacity planning, and early "
            "recognition of system pressure. The released benchmark contains 96 related but "
            "distinct hourly series, 22 explanatory variables, and one nonnegative target. "
            "Future covariates are supplied, whereas future target values are hidden. The "
            "model must therefore combine daily and weekly regularities with unit-specific "
            "dynamics and external signals while limiting error accumulation over fourteen "
            "recursive blocks.", s["body"]),
        Paragraph(
            "We favor a small recurrent model over a much larger architecture so controlled "
            "ablations and CPU-only evaluation remain practical. The work contributes a "
            "seasonally anchored residual LSTM, leakage-safe preprocessing, exact recursive "
            "inference, and a two-target transfer experiment on Jena climate data.", s["body"]),
        Paragraph("Problem contract", s["h2"]),
        results_table(
            [["Frequency", "History", "Block", "Required horizon", "Metric"],
             ["Hourly", "168 h", "24 h", "336 h", "WAPE"]],
            [31 * mm, 31 * mm, 31 * mm, 39 * mm, 31 * mm],
        ),
    ]

    story += [PageBreak(), Paragraph("2. Related Work", s["h1"]),
        Paragraph(
            "Long short-term memory networks use gated recurrent updates to preserve useful "
            "information over long sequences [1]. Recurrent architectures remain widely used "
            "for nonlinear sequential modeling [2], including robust time-series prediction "
            "[3]. Attention-based recurrent models can separate feature selection and temporal "
            "attention [4], but the present data volume and reproducibility requirements favor "
            "a smaller model.", s["body"]),
        Paragraph(
            "Seasonal persistence is unusually competitive in short-horizon forecasting. "
            "Instead of asking a neural network to reconstruct the entire weekly level, we use "
            "a seasonal mean as an explicit inductive bias and learn only deviations explained "
            "by current load, risk, capacity, and calendar variables.", s["body"]),
        Paragraph("3. Method", s["h1"]),
        Paragraph("3.1 Leakage-safe preprocessing", s["h2"]),
        Paragraph(
            "Rows are sorted per series and checked for duplicate timestamps and hourly "
            "spacing. The final 336 public training hours are reserved for selection. "
            "Continuous covariates are standardized with statistics from the remaining interval "
            "only; binary and cyclic variables are unchanged. Missing inputs receive an "
            "indicator, are forward-filled within their series, and use the training median "
            "only if no earlier value exists.", s["body"]),
        Paragraph(
            "A training window contains 168 history hours and 24 output hours. Starts are six "
            "hours apart and occur only after four complete weeks. Historical targets are "
            "standardized per series for the encoder, while outputs and the L1 loss remain on "
            "the original scale.", s["body"]),
        Paragraph("3.2 Architecture", s["h2"]),
        ArchitectureDiagram(document.width, 72 * mm),
        Paragraph(
            "Figure 1. The encoder state, future covariates, and unit embedding drive a shared "
            "residual decoder. The residual is added to the seasonal reference.", s["caption"]),
    ]

    story += [PageBreak(), Paragraph("3.3 Optimization and recursive inference", s["h2"]),
        Paragraph(
            "The encoder has one LSTM layer with 64 hidden units. Its final state is processed "
            "by Layer Normalization and concatenated with an eight-dimensional series embedding "
            "and known future covariates. A two-layer decoder emits 24 residuals. The reference "
            "averages matching hours from the previous four weeks, falling back to weekly and "
            "then last-value persistence when history is short. Predictions are clamped to zero "
            "only during inference.", s["body"]),
        Paragraph(
            "We optimize raw-scale L1 with AdamW, learning rate 0.001, weight decay 0.0001, "
            "batch size 256, and gradient clipping at 1.0. Training uses seed 42, at most 30 "
            "epochs, and patience five. Each predicted block is appended to history before the "
            "next one, exactly matching private evaluation.", s["body"]),
        Paragraph("4. Benchmark Experiments", s["h1"]),
        Paragraph("4.1 Protocol", s["h2"]),
        Paragraph(
            "The internal holdout contains the last 336 hours for every unit. No held-out target "
            "is exposed during preprocessing, training, or rollout. WAPE is aggregated across "
            "all units and timestamps. Once the full model was selected, it was retrained from "
            "initialization on all public targets for the selected 12 epochs.", s["body"]),
        results_table(
            [["Method", "Internal WAPE", "Compared with last value"],
             ["Last value", "0.545040", "reference"],
             ["Daily persistence", "0.418006", "-23.3%"],
             ["Weekly persistence", "0.454876", "-16.5%"],
             ["Four-week seasonal mean", "0.341780", "-37.3%"],
             ["Target-history-only LSTM", "0.383264", "-29.7%"],
             ["Full multivariate LSTM", "0.241293", "-55.7%"]],
            [75 * mm, 42 * mm, 49 * mm],
        ),
        Paragraph("Table 1. Chronological benchmark results; lower WAPE is better.", s["caption"]),
        Paragraph("4.2 Ablation and interpretation", s["h2"]),
        Paragraph(
            "The target-only LSTM beats last-value, daily, and weekly persistence but not the "
            "four-week seasonal mean. The full model reduces WAPE by 29.4% relative to that "
            "strongest baseline. Since both neural variants share the seasonal anchor, encoder, "
            "decoder, and optimizer, the difference is direct evidence that the released "
            "covariates provide information beyond target history. Validation improved through "
            "epoch 12; later training-loss reductions did not generalize.", s["body"]),
        Paragraph(
            "The final CSV contains exactly 32,256 finite, nonnegative predictions in forecast "
            "index order. The packed checkpoint stores feature ordering, imputation statistics, "
            "scalers, series mapping, recent context, architecture settings, and weights.", s["body"]),
    ]

    story += [PageBreak(), Paragraph("5. Additional Dataset: Jena Weather", s["h1"]),
        Paragraph(
            "The Jena climate record from the Max Planck Institute for Biogeochemistry contains "
            "ten-minute meteorological observations from 2009 through 2016 [5]. We aggregate "
            "numerical variables to hourly means. Negative sentinel wind speeds are marked "
            "missing before aggregation, and wind direction is converted to Cartesian x/y "
            "components. Air temperature and relative humidity are predicted jointly.", s["body"]),
        Paragraph("5.1 Experimental design", s["h2"]),
        Paragraph(
            "Calendar years 2009-2014 form the training split, 2015 is validation, and 2016 is "
            "test. At each daily origin, 168 observed hours predict the next 24. Historical "
            "meteorological measurements enter the encoder; only hour- and year-cycle sine/cosine "
            "features are available for future timestamps. This prevents access to future measured "
            "weather. The model selected epoch 12 by mean normalized validation MAE.", s["body"]),
        results_table(
            [["Method", "Temp. MAE", "Temp. RMSE", "Humidity MAE", "Humidity RMSE", "Norm. MAE"],
             ["Last value", "2.719", "3.712", "11.169", "15.126", "0.501"],
             ["Daily persistence", "2.432", "3.202", "9.884", "12.991", "0.445"],
             ["Weekly persistence", "4.565", "5.704", "12.157", "15.522", "0.639"],
             ["Residual LSTM", "2.706", "3.442", "8.903", "11.745", "0.431"]],
            [43 * mm, 24 * mm, 25 * mm, 29 * mm, 30 * mm, 22 * mm],
        ),
        Paragraph("Table 2. Jena 2016 test results across 366 non-overlapping forecast origins.", s["caption"]),
        Paragraph("5.2 Transfer analysis", s["h2"]),
        Paragraph(
            "The LSTM has the best combined normalized MAE and improves humidity MAE by 10% "
            "relative to daily persistence. Daily persistence remains 0.275 degrees C better "
            "for temperature MAE, showing that no single method dominates each channel. The "
            "model transfers successfully at the architecture level: only the number of series, "
            "output channels, historical measurements, and future calendar inputs change.", s["body"]),
        Paragraph(
            "Weather also exposes a limitation of a fixed seasonal anchor. Temperature has a "
            "strong daily cycle, while humidity reacts to fronts and precipitation that are not "
            "known at the forecast origin. The multivariate encoder helps humidity most, but a "
            "future experiment could add numerical weather predictions or target-specific loss "
            "weights.", s["body"]),
        Paragraph("Jena result: best combined error and best humidity forecast", s["callout"]),
    ]

    story += [PageBreak(), Paragraph("6. Relation to the Expose", s["h1"]),
        Paragraph(
            "The final method follows the expose's selection of an LSTM and its planned daily "
            "and weekly baselines. Inspection of the released data required several corrections. "
            "The benchmark is operational rather than meteorological, so its known load, risk, "
            "capacity, and calendar covariates are modeled directly. Jena measurements occur "
            "every ten minutes rather than every twenty. We replaced the tentative percentage "
            "split with reproducible calendar-year boundaries and made the weather objective "
            "explicit: joint temperature and humidity prediction. AR, MA, and ARMA remain related "
            "comparison ideas rather than implemented models.", s["body"]),
        Paragraph("7. Limitations", s["h1"]),
        Paragraph(
            "The benchmark analysis uses one temporal holdout and one initialization, so it does "
            "not measure variance across cutoffs or random seeds. Recursive errors may grow under "
            "distribution shift, and the seasonal reference assumes stable weekly structure. The "
            "compact LSTM cannot represent every long-range interaction available to larger "
            "attention or state-space models. The Jena experiment uses observed measurements only "
            "in history, making rapidly changing weather inherently difficult.", s["body"]),
        Paragraph("8. Conclusion", s["h1"]),
        Paragraph(
            "A small residual LSTM substantially outperforms strong persistence rules on the "
            "operational benchmark while remaining easy to reproduce and package. The target-only "
            "ablation demonstrates that the main gain depends on multivariate information rather "
            "than neural capacity alone. On Jena, the architecture achieves the best combined "
            "temperature/humidity error and materially improves humidity forecasting. The final "
            "archive runs offline on CPU and reproduces the evaluator's exact command.", s["body"]),
        Paragraph("Contributions", s["h2"]),
        Paragraph(
            "<b>Required author check before submission:</b> Tudor Orza, Marie Reinhold, and Felix "
            "Besser must replace this note with their agreed individual contributions. No "
            "allocation was invented because it cannot be inferred reliably from the code or expose.",
            s["abstract"]),
        Paragraph("Reproducibility checklist", s["h2"]),
        results_table(
            [["Artifact", "Status"],
             ["Deterministic training configuration", "Complete"],
             ["Benchmark and Jena result JSON", "Complete"],
             ["Exact CPU inference interface", "Verified"],
             ["Self-contained model archive", "Verified"],
             ["Hugging Face validation score", "Pending external submission"],
             ["Individual contribution statement", "Pending group confirmation"]],
            [95 * mm, 70 * mm],
        ),
    ]

    refs = [
        "[1] S. Hochreiter and J. Schmidhuber. Long Short-Term Memory. Neural Computation 9(8), 1735-1780, 1997. doi:10.1162/neco.1997.9.8.1735.",
        "[2] I. D. Mienye, T. G. Swart, and G. Obaido. Recurrent Neural Networks: A Comprehensive Review of Architectures, Variants, and Applications. Information 15(9), 517, 2024. doi:10.3390/info15090517.",
        "[3] J. T. Connor, R. D. Martin, and L. E. Atlas. Recurrent Neural Networks and Robust Time Series Prediction. IEEE Transactions on Neural Networks 5(2), 240-254, 1994. doi:10.1109/72.279188.",
        "[4] Y. Qin, D. Song, H. Chen, W. Cheng, G. Jiang, and G. Cottrell. A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction. IJCAI, 2627-2633, 2017. doi:10.24963/ijcai.2017/366.",
        "[5] Max Planck Institute for Biogeochemistry. Weather Data from the Max Planck Institute for Biogeochemistry in Jena. https://www.bgc-jena.mpg.de/wetter/ (accessed 2026-09-04).",
    ]
    story += [PageBreak(), Paragraph("References", s["h1"])]
    story += [Paragraph(reference, s["reference"]) for reference in refs]
    story += [Spacer(1, 8 * mm), Paragraph(
        "The report body occupies five pages; this reference page is excluded from the assigned "
        "4-6 page limit.", s["small"])]
    document.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    print(f"Created {output}")


if __name__ == "__main__":
    build(Path("output/pdf/final_report.pdf"))
