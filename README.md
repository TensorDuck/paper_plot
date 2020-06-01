# paper_plot

Plot figures in scientific publications/presentations in a consistent format.

## Background

During my PhD work, I would routinely have a copy of all the raw data I was
using for a paper inside a single directory. I would then have a plotting
package alongside of it in. Mainly:
* Made it quick to regenerate and revise figure formatting based on the exact
  raw data I had collected.
* Figures could easily be re-formatted into presentation versus paper format for
  presenting at conferences/talks.
* Made it easy to work alongside a tex file that referenced the files.

In short, I found it time consuming and difficult to use a specialized software
(i.e. Origin) to manually edit figures. Such software is also generally not
free. Thus, I made a programmatic way to edit and format figures was made using
`matplotlib`.

Several of my group members also found it useful, and now I am sharing a cleaned
up version of the code for others (mainly academics) to use.


## Usage

The main class is in `paper_plot/plotter.py`. I recommend either:
* Sub-classing the method and extending it to your specific figure needs.
* Copy and paste the code and modify to your needs.

The code likely won't be perfect for your needs out of the box, but hopefully
the paradigm I used will be of use to mimic for others.


## License

See LICENSE for a full description of how this repo can be used. But generally,
you should feel free to have at it.