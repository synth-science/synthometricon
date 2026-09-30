# Treemap helper, vendored from github.com/rubenarslan/construct_proliferation
# (0_functions.R) so this folder is self-contained.
# `tests` needs columns DOI, shortName, Name, parent, n. `colors[1]` is the
# background colour; the remaining colours are recycled over the tiles.

treemap_graph <- function(tests, colors = c("#69D2E7", "#A7DBD8", "#E0E4CC", "#F38630", "#FA6900")) {
  bgcolor <- colors[1]
  colors <- colors[-1]
  fig <- plotly::plot_ly(
    type = "treemap",
    labels = stringr::str_c('<a href="https://dx.doi.org/', tests$DOI, '">', tests$shortName, "</a>"),
    parents = tests$parent,
    values = tests$n,
    text = stringr::str_c(dplyr::if_else(tests$Name == tests$shortName, "", stringr::str_c(tests$Name, "<br>")), tests$DOI),
    marker = list(
      line = list(width = 0.1, color = "white"),
      colors = rep(colors, length.out = nrow(tests))
    ),
    tiling = list(
      pad = 0,
      packing = "squarify",
      squarifyratio = (1 + sqrt(5)) / 2
    ),
    hoverinfo = "label+value+text+percent root",
    textinfo = "label"
  ) %>%
    plotly::config(displaylogo = FALSE, displayModeBar = FALSE)

  fig %>% plotly::layout(
    autosize = TRUE,
    paper_bgcolor = "white",
    plot_bgcolor = bgcolor,
    uniformtext = list(minsize = 15, mode = "hide"),
    margin = list(l = 0, t = 0, r = 0, b = 0)
  )
}
