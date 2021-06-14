%% LaTeX2e class for student theses
%% thesis.tex
%% 
%% Based on SDQ KIT Template by Erik Burger
%%
%% Karlsruhe Institute of Technology
%% Institute for Automation and Applied Informatics
%% AIDA Research Group
%%
%% Nicole Ludwig
%% nicole.ludwig@kit.edu
%%
%% Version 1.2.1, 2018-10-11


\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{thesisIAI}[2018/06/05 v1.0 Student thesis class]
\DeclareOption{ngerman}{\PassOptionsToPackage{\CurrentOption}{babel}}
% German is also needed in English thesis for the abstract
\DeclareOption{english}{\PassOptionsToPackage{main=\CurrentOption,ngerman}{babel}}
\ProcessOptions\relax

%% ---------------------------------------------------------------
%% | Based on KOMAscript by Markus Kohm http://www.komascript.de |
%% ---------------------------------------------------------------
\LoadClass[12pt,twoside,headsepline,a4paper,titlepage=true,DIV=15,BCOR=20mm]{scrbook}

%% -----------------------
%% | Headers and footers |
%% -----------------------
% draft=false removes rulers from header and footer
\RequirePackage[automark, draft=false]{scrlayer-scrpage}

\clubpenalty=10000 %prevent orphans
\widowpenalty=10000 %prevent widows
\displaywidowpenalty = 10000

%%% Diverse Pakete und Einstellungen =====================================
 
%%% Doc: ftp://tug.ctan.org/pub/tex-archive/macros/latex/contrib/xcolor/xcolor.pdf
% Farben
% Incompatible: Do not load when using pstricks !
\usepackage[
  dvipsnames,
  table % Load for using rowcolors command in tables
]{xcolor}

%%% Doc: Documentation inside dtx file
% Mehrere Text-Spalten
\usepackage{multicol}

\usepackage{lscape}
\usepackage{rotating}
 
\nonfrenchspacing     % liefert extra Platz hinter Satzenden.
                       % Fuer deutschen Text standardmaessig ausgeschaltet!
 
\usepackage{ellipsis}  % >>Intelligente<< \dots


%% Zeilenabstand =========================================================
%
%%% Doc: ftp://tug.ctan.org/pub/tex-archive/macros/latex/contrib/setspace/setspace.sty
\usepackage{setspace}
%\doublespace          % 2-facher Abstand
\onehalfspace        % 1,5-facher Abstand
% hereafter load 'typearea' again

%% --------------
%% | Typography |
%% --------------

% T1 font encoding
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}

% serif type: Linux Libertine
\RequirePackage{libertine}
% Linux Libertine in math mode
\RequirePackage[libertine]{newtxmath}

% grotesque type: Adobe Source Sans Pro
\RequirePackage[scaled=.92]{sourcesanspro}

% monospace type: Bera Mono
\RequirePackage[scaled=.78]{beramono}

% semi-bold type (for subsections and paragraphs)
\newcommand*{\sbdefault}{sb}
\DeclareRobustCommand{\sbseries}{%
  \not@math@alphabet\sbseries\relax
  \fontseries\sbdefault\selectfont}

\DeclareTextFontCommand{\textsb}{\sbseries}

\addtokomafont{subsection}{\sbseries}
\addtokomafont{subsubsection}{\sbseries}
\addtokomafont{paragraph}{\sbseries}

% microtype for nicer typography
\RequirePackage[protrusion=true,expansion=true]{microtype}

% commands for code and model keywords in text
\newcommand{\code}[1]{\texttt{\hyphenchar\font45\relax #1}}
\newcommand{\model}[1]{\textsf{#1}}

% nicer spacing for enumerations
\RequirePackage{enumitem}

\setlength{\headheight}{1.1\baselineskip}

%% --------------
%% | Title page |
%% --------------

% Logos
\newcommand{\thegrouplogo}{sdqlogo}
\newcommand{\grouplogo}[1]{\renewcommand{\thegrouplogo}{#1}}
\newcommand{\printgrouplogo}{\includegraphics[height=1.8cm]{logos/\thegrouplogo}}
\newcommand{\nogrouplogo}{\renewcommand{\printgrouplogo}{}}

\titlehead{%
% KIT logo german/english
\iflanguage{ngerman}
{\begin{center}
\includegraphics[height=2.5cm]{logos/kitlogo_de_cmyk}
\end{center}}
{\begin{center}
\includegraphics[height=2.5cm]{logos/kitlogo_en_cmyk}
\end{center}}
%\hfill
%% SDQ logo
%\printgrouplogo
\ThisCenterWallPaper{1}{title-background.pdf}
}
		
%
\setkomafont{title}{\huge\sffamily\bfseries}
\setkomafont{subtitle}{\normalfont\large}

\newcommand{\thefaculty}{}
\newcommand{\faculty}[1]{\renewcommand{\thefaculty}{#1}}

%% variables for title page
\newcommand{\theinstitute}{\iflanguage{english}
    {Institute for Automation and Applied Informatics (IAI)}
    {Institut für Automation und angewandte Informatik (IAI)}}
\newcommand{\thethesistype}{}
\newcommand{\thereviewerone}{}
\newcommand{\thereviewertwo}{}
\newcommand{\theadvisorone}{}
\newcommand{\theadvisortwo}{}
\newcommand{\theeditstart}{}
\newcommand{\theeditend}{}
\newcommand{\thefacultyname}{}

%% formatting commands for titlepage
\newcommand{\thesistype}[1]{\subtitle{\vskip2em #1 \iflanguage{english}{of}{von}}%
\renewcommand{\thethesistype}{#1}}
\newcommand{\myinstitute}[1]{\renewcommand{\theinstitute}{#1}}
\newcommand{\reviewerone}[1]{\renewcommand{\thereviewerone}{#1}}
\newcommand{\reviewertwo}[1]{\renewcommand{\thereviewertwo}{#1}}
\newcommand{\advisorone}[1]{\renewcommand{\theadvisorone}{#1}}
\newcommand{\advisortwo}[1]{\renewcommand{\theadvisortwo}{#1}}

\newcommand{\facultyname}[1]{\renewcommand{\thefacultyname}{#1}}

\newcommand{\editingtime}[2]{%
\renewcommand{\theeditstart}{#1}%
\renewcommand{\theeditend}{#2}%
%% do not show the date
\date{}
}

\newcommand{\settitle}{%
\publishers{%
\large
\thefacultyname \\
\theinstitute\\[2em]
\begin{tabular}{l l}
  \iflanguage{english}{Reviewer}{Erstgutachter}: & \thereviewerone\\
	\ifthenelse{\equal{\thereviewertwo}{}}{}{%
  \iflanguage{english}{Second reviewer}{Zweitgutachter}: & \thereviewertwo\\
	}
	% if there is no advisor, do not output this line
	\ifthenelse{\equal{\theadvisorone}{}}{}{%
  \iflanguage{english}{Advisor}{Betreuender Mitarbeiter}: &  \theadvisorone\\
	}
  % if there is no second advisor, do not output this line
  \ifthenelse{\equal{\theadvisortwo}{}}{}{%
  \iflanguage{english}{Second advisor}{Zweiter betreuender Mitarbeiter}: & \theadvisortwo\\
  }
\end{tabular}
\vskip2em
\theeditstart{} -- \theeditend
}
}

%% -----------------------------
%% | Abstract/Acknowledgements |
%% -----------------------------

\newcommand{\abstract}[1][\abstractname]{\chapter*{#1}}
\newcommand{\Abstract}[1][\abstractname]{\chapter*{#1}\addcontentsline{toc}{chapter}{#1}} 

\def\ackname{Acknowledgments}
\def\abstractname{Abstract}
\def\switcht@deutsch{\svlanginfo
	\def\ackname{Danksagung}
	\def\abstractname{Kurzfassung}
}
\def\switcht@english{\svlanginfo
	\def\ackname{Acknowledgements}
	\def\abstractname{Abstract}
}

%%% In English theses, an additional German Abstract is needed.
%\newcommand{\includeabstract}{
%\iflanguage{english}{
%% include English and German abstracts
%\include{sections/abstract_en}
%\begin{otherlanguage}{ngerman}
%\include{sections/abstract_de}
%\end{otherlanguage}
%}{
%% include only German abstract
%\include{sections/abstract_de}
%}
%}

%% ------------
%% | Packages |
%% ------------

% draft mode
\RequirePackage{ifdraft}
\RequirePackage{ifthen}

% enumerate subsubsections
\setcounter{secnumdepth}{3}

% wallpaper for title page (KIT-Frame)
\RequirePackage{wallpaper}

% subimport of files
\RequirePackage{import}

% languages
\RequirePackage{babel}

% hyphenation for compound words
\RequirePackage[shortcuts]{extdash}

% nice tables
\RequirePackage{booktabs}
\RequirePackage{longtable}
\RequirePackage{array}

% show graphics in draft mode
\RequirePackage{graphicx}
\setkeys{Gin}{draft=false}

% % Mathematics package
\RequirePackage{amsmath}
\let\openbox\relax
\RequirePackage{amsthm}
%\RequirePackage{amssymb}
\RequirePackage{mathtools}

% appendix
\RequirePackage[toc,title,header]{appendix}
\noappendicestocpagenum

% todo-notes
\RequirePackage[obeyDraft]{todonotes}
\presetkeys{todonotes}{inline}{}

% PDF specific packages
\RequirePackage[hyphens]{url}
\RequirePackage[breaklinks,colorlinks=false, draft=false]{hyperref}
\newcommand\setpdf{\hypersetup{%
    pdftitle={\thethesistype},%
    pdfsubject={\@title},%
    pdfauthor={\@author},%
    pdfborder={0 0 0},%
    %linkcolor=kit-blue100,%
    %citecolor=kit-green100,%
    %urlcolor=kit-red100
    }%
    \let\theauthor\@author
}

%% automatic naming of sections with autoref
\def\sectionautorefname{Section}
\def\subsectionautorefname{Subsection}
\def\subsubsectionautorefname{Subsection}
\def\paragraphautorefname{Paragraph}

