Reporting a problem
###################

ScanTailor writes a log file every time it runs. When something goes wrong, the
log usually says what happened: which page and stage failed, which versions were
running, and whether the previous session crashed. Attach it to your report on
the `issue tracker <https://github.com/adehad/py-scantailor/issues/new/choose>`_.

Finding the log
***************

Choose :menuselection:`Help --> Open log folder` to open the folder in your file
manager. The folder is:

.. list-table::
   :header-rows: 1

   * - System
     - Log folder
   * - Windows
     - ``%LOCALAPPDATA%\ScanTailor\logs``
   * - macOS
     - ``~/Library/Logs/ScanTailor``
   * - Linux
     - ``$XDG_STATE_HOME/scantailor/logs``, which is
       ``~/.local/state/scantailor/logs`` unless you set ``XDG_STATE_HOME``

Which files to attach
*********************

.. list-table::
   :header-rows: 1

   * - File
     - What it holds
   * - ``gui-<date>-<time>-<process>.log``
     - One window session. Attach the newest one from the day of the problem.
   * - ``crash-gui-<date>-<time>-<process>.log``
     - Left behind only when ScanTailor crashed. Attach any with the same date
       and time as the session log.
   * - ``crash-worker-<date>-<time>-<process>.log``
     - Left behind when a worker process that processes pages crashed. The
       session log then has a line ``worker process died`` naming the pages
       that were running. Attach the non-empty ones with the same date as the
       session log.
   * - ``cli-<date>-<time>-<process>.log``
     - One ``scantailor-cli run`` or ``export``. Attach it for a problem with
       the command line.

When ScanTailor starts after a session that did not close normally, its log
names the earlier file in a line ``previous session did not end cleanly``. That
earlier file is the one to attach.

ScanTailor keeps the newest ten files of each kind and at most 100 MB in total,
and deletes older ones when it starts. A single session's log is capped at about
20 MB.

Before you attach a log
***********************

Logs stay on your computer until you send them. They contain the full paths of
your projects and scans, so folder names, your user name and book titles appear
in them. Open the file in a text editor and check it before attaching it. You
can replace names you do not want to share; keep the rest of each line.

If one page comes out wrong, the log shows what ScanTailor decided, but the scan
itself is often what makes the problem reproducible. Attach that page too if you
can share it.

The command line
****************

``scantailor-cli run`` and ``scantailor-cli export`` write a log in the same
folder. Other commands write one only with ``--log-file PATH``, which also
works for ``run`` and ``export`` to put the log somewhere else, for example in a
CI job's artefacts. When a command fails unexpectedly, its error message names
the log file.
