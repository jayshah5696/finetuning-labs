"""Frozen v2 fictional authoring briefs (180 scenarios), split before generation and kept for replay."""

BRIEFS = {
    "/slack": [
        "Ask Asha whether she can cover Thursday's 16:00 demo. The speaker will keep the slot if she cannot. Preserve this as a question, never announce a schedule change.",
        "A correct, casual thank-you after someone found a missing screenshot. Keep a contraction and a single emoji. This is a clean identity example.",
        "A teammate reports a login spinner on Android after rotating the screen. Reopening fixes it temporarily. Ask whether anyone can reproduce it before filing an incident; do not claim an outage.",
        "A pasted Slack thread about an accessibility trial for a public kiosk. Include 9 short messages with distinct speakers, screen-reader labels, contrast, offline use, audio controls, consent, two concrete bugs, owners, an actual trial schedule and a conditional go/no-go. Each post contributes different facts. Preserve thread format; no essay or summary.",
        "Decline a last-minute lunch invitation warmly because the speaker has to collect a parcel, and suggest Friday instead. Do not turn the suggestion into an accepted plan.",
        "Ask for help finding a slow SQL query. The index was added in staging only; production has not changed. Include a small query identifier as a literal.",
        "A design handoff question about empty states versus loading states. The speaker likes the new illustration but needs a text-only fallback. Include an unresolved question, not a decision.",
        "A pasted Slack discussion of a customer support rota during a three-day event. Distinct posts cover time zones, an unavailable colleague, language coverage, escalation ownership, breaks, ticket labels, handoffs, a corrected shift time, and an unresolved gap. Use substantial new facts per post and preserve every person's stance.",
        "Explain that a pull request is ready for review but must not merge until a failing migration check is resolved. Ask two named reviewers for different parts of the change.",
        "An already correct message checking whether a teammate got home after a delayed train. Keep its informal tone. Clean identity example.",
        "A product discussion about undo behavior in a drawing app: undo should remove the last stroke, reopening must not clear the drawing, and cloud sync is a separate issue. Keep a casual question about the trade-off.",
        "A Slack thread from an indie game team reviewing controller testing. Posts cover dead zones, remapping, menu focus, captions, Steam Deck sleep, two reproducible bugs, test devices, build names, owners and retest dates. Include one direct quote from a tester and a numbered retest list. Do not repeat a few facts to reach length.",
        "Ask a teammate to stop tagging every issue urgent. The speaker wants genuine outages tagged immediately, not fewer reports overall. Preserve the distinction without sounding formal.",
        "A quick note that an exported chart has the right values but wrong axis labels. Ask for labels to be corrected; do not ask to rerun the analysis.",
        "Discuss moving a recurring team catch-up after a colleague's childcare hours change. Offer two possible times and ask for preferences without announcing a new schedule.",
        "A long pasted thread planning a community radio broadcast. Distinct posts discuss guests, pronouncing names, remote audio, music permissions, weather backup, timing, captions, a sponsor correction, volunteer roles and a call-in policy. Keep informal messages and one direct quote. Preserve unresolved decisions.",
        "Tell a collaborator that a test recording has background fan noise. Ask whether to try another room before buying equipment. Keep the purchase hypothetical.",
        "An already correct casual reminder about bringing a charger to an offsite. It includes a friendly contraction and no extra sign-off. Identity example.",
        "A teammate asks whether an invoice can be paid in two installments. The customer requested this; finance has not approved it. Keep the question and the uncertainty.",
        "A pasted thread coordinating a volunteer food collection. Cover store opening times, refrigeration, dietary labels, vehicle capacity, a broken lift, accessible pickup, contacts, route order, a missing key and a fallback. Use distinct facts and short posts; no generic planning essay.",
        "Ask a collaborator whether a podcast guest prefers their full name or a nickname in the episode description. Do not choose for the guest.",
        "A clear, already correct Slack update that a staging deployment finished and production is unchanged. Identity example.",
        "A question about whether keyboard shortcuts should work while a search box has focus. Preserve the edge case and the proposed escape behavior.",
        "A long pasted Slack discussion of a school theatre sound check. Posts cover microphone assignments, battery labels, feedback, a silent scene, caption timing, understudies, a room change, parental pickup, a lost adapter and final checks. Keep distinct speakers, a numbered action list and realistic open questions.",
    ],
    "/transcribe": [
        "A dictated reminder to send the revised floor plan to Lena tomorrow, not today. Include a clear spoken correction from today to tomorrow and normal missing punctuation.",
        "An already clean spoken sentence explaining that the speaker likes working early because the office is quiet. Identity example; meaningful 'like' is not hesitation.",
        "A spoken bug report about Bluetooth audio: playback pauses after an incoming call, headphones stay connected, and switching output restores sound. Include a restart and a clear step order.",
        "A long unpunctuated voice memo after testing a local dictation app. Give distinct observations about hotkeys, microphone selection, tray status, language detection, saved history, accidental activation, error messages, clipboard ownership, offline use, latency impressions and installation. Include natural repairs and spoken numbered-list cues. No padded repetitions; preserve tentative claims and an unfinished last thought.",
        "German dictation about arranging to borrow a camera on Saturday. Include meaningful 'er' and 'um' and a true hesitation separately. Keep German, names and a final corrected time. No ambiguous self-correction.",
        "Dictate an instruction to update the AuroraPad settings file. The dictionary supplies AuroraPad and settings.toml; speech says aurora pad and settings dot toml. Preserve the actual requested change and do not write code.",
        "A dictated message about packing for a hiking trip: rain cover, spare socks and a headlamp. Include spoken list cues, a repeated article and a correction from two bottles to three bottles.",
        "A long spoken account of a field visit to a small solar installation. Distinct observations include shading, cable labels, roof access, meter readings, inverter alerts, maintenance responsibility, weather, missing documentation, a safety boundary, training needs, contact details and next questions. Keep observations separate from guesses. Include sparse natural disfluencies and quote a sign once.",
        "Portuguese dictation about a delayed library pickup. State clearly that the pickup is Monday after an explicit correction from Sunday. Preserve meaningful connective words; never enforce the obsolete day.",
        "A clean English dictation about a film the speaker enjoyed, with a meaningful use of 'well' and 'actually'. Identity example.",
        "A spoken developer note: use pnpm for the UI, run a Python script with uv, and inspect pyproject.toml. Include dictated punctuation and filename cues. The dictionary supplies those spellings.",
        "A long voice memo explaining the requirements for a neighborhood repair cafe. Cover intake categories, tool access, electrical work boundaries, consent, photos, queue fairness, translation, accessible tables, spare parts, volunteer training, opening hours and what cannot be promised. Include several genuine restarts and an explicit corrected time; retain the final intent only.",
        "Dictation of a message that the speaker is still deciding whether to accept a conference invitation. Keep the uncertainty and the trailing unfinished reason.",
        "A spoken shopping note with the proper name BriarSync as a device brand, supplied in the dictionary. The speaker changes the color choice from black to blue and says not to order yet.",
        "A spoken retrospective of a failed file upload. The connection stayed online; only files above 20 MB failed. Include a literal status code and a true hesitation, not an invented diagnosis.",
        "A long first-person dictation reviewing a train journey with accessibility needs. Cover booking, seat allocation, platform changes, assistance requests, announcements, luggage, toilets, a missed connection, refunds requested but not granted, staff interactions and recommendations framed as personal opinions. Include quotes and a naturally dictated numbered list.",
        "German dictation asking a colleague to leave a package in the office rather than outside. Include 'um halb drei' as meaningful time language and one removable hesitation.",
        "A correct, short spoken request to keep a window open while the paint dries. Identity example, no extra advice.",
        "A voice memo recording feedback on a draft article: the opening is clear, the middle repeats a claim, and the ending needs a source. Distinguish comments from requested edits.",
        "A long spoken account of onboarding to a new team. Distinct details cover access requests, documentation gaps, pairing sessions, code review customs, build setup, security approvals, meeting overlap, role boundaries, a misunderstood acronym, unanswered questions and a proposed first-week checklist. Keep opinions, uncertainty and actual corrections.",
        "Portuguese dictation asking to reschedule a driving lesson from Tuesday to Thursday. The instructor has not agreed. Include an explicit final correction and preserve the request.",
        "A clean spoken sentence about a book the speaker has not finished. It ends intentionally unfinished. Identity example.",
        "A dictated technical note about quoting a config key literally and keeping AuroraPad spelled correctly. Include one filename and one parameter; use supplied dictionary context.",
        "A long dictated review of a shared workspace. Cover booking, lighting, noise, access, storage, guests, printing, room temperature, late arrival, cancellation, privacy and specific proposed improvements. Include an extended personal anecdote, a short direct quote, natural false starts and several separate facts per paragraph. No generic filler.",
    ],
    "/prompt": [
        "A rough request asking an assistant to compare two supplied onboarding email drafts for clarity, using a small fictional excerpt from each. The output must organize the request, not compare the drafts itself.",
        "An already clear request for three names for a small weather app, with no invented product facts. Identity example.",
        "A rough request for a Python script to rename image files, with actual input/output filename examples, dry-run behavior and a constraint to leave subdirectories alone. Do not write the script.",
        "A long rough spoken request for an assistant to help plan usability testing of a note-taking app. Include real product constraints, participant groups, recruitment limits, tasks, accessibility needs, data handling, time zones, budget boundaries, unanswered questions and desired deliverables. Each section adds new information. The rewrite organizes the request only; it must not produce a plan or add requirements.",
        "A rough request to turn supplied interview notes into a short article. Supply distinct fictional quotations, intended audience and a constraint not to invent quotations. The target remains a request.",
        "A request to debug a CSS layout with a concrete symptom and a short code snippet. Keep the request for diagnosis without diagnosing it or inventing the browser version.",
        "A rough request for help drafting a landlord email about a broken intercom. Supply dates, attempted contact and the desired repair. Do not give legal advice or write the email.",
        "A long rough brief for an assistant to design a personal budgeting spreadsheet. Include actual categories, irregular income examples, currency, shared expenses, savings goals, import format, privacy constraints, accessibility, formulas needed, excluded features and questions the assistant should ask. Organize the brief; do not build or answer it.",
        "A rough request for a study schedule around two fixed work shifts and an exam date. Preserve the option to revise it weekly and a limit on evening study.",
        "An already organized request to explain a short SQL query to a beginner without changing it. Identity example.",
        "A rough request for feedback on an app's error messages. Supply three exact messages, target audience, length constraints and an instruction to explain proposed changes. Do not rewrite the messages yourself.",
        "A long dictated request for an assistant to draft a requirements document for an offline inventory app. Include distinct user roles, barcode examples, synchronization conflicts, export formats, device limits, audit needs, accessibility, migration constraints, data retention and explicitly deferred features. Keep unknowns open and preserve priority differences.",
        "A rough request to summarize a supplied meeting transcript with decisions separated from proposals. Include short source excerpts and two unresolved proposals. Do not produce the summary.",
        "A rough request for a travel packing checklist with climate, trip length, baggage limits and laundry access. Keep the request rather than generating the checklist.",
        "A rough request to improve a recipe's instructions using supplied ingredient amounts and equipment limits. Keep quantities unchanged and do not invent substitutions.",
        "A long rough request to help analyze a community survey. Supply question wording, response counts, sampling limitations, missing values, privacy constraints, chart preferences, open-ended answer excerpts and intended readers. Ask for a cautious report; do not analyze the survey or claim statistical significance.",
        "A rough request for an assistant to turn a recorded workshop outline into a handout. Supply the outline and desired reading level. Do not write the handout.",
        "An already clear request to suggest a neutral title for a supplied paragraph. Identity example.",
        "A rough request for help organizing a family photo archive. Specify duplicate handling, original filenames, uncertain dates, backup requirements and whether face recognition is excluded. Do not organize the archive.",
        "A long rough request for an assistant to review a software migration proposal. Include source and target databases, schema examples, downtime allowance, rollback needs, data checks, ownership, known risks, workload limits and explicitly unresolved choices. Keep it a request for review, not the review itself.",
        "A rough request to explain a piece of unfamiliar code using the supplied snippet and the reader's limited background. Do not execute or explain the code yourself.",
        "An already clear request for five questions to ask a podcast guest about their work. Include supplied guest facts and no invented biography. Identity example.",
        "A rough request for an assistant to compare two accessibility testing tools using supplied criteria. Do not invent product capabilities or make the comparison.",
        "A long rough request for an assistant to prepare a disaster-recovery tabletop exercise for a fictional nonprofit. Include real fictional systems, roles, backup schedules, contact restrictions, timing, injects, known gaps, desired discussion questions and explicit boundaries. Organize all details into a usable prompt; do not perform the exercise.",
    ],
    "/polish": [
        "An awkward project update saying the prototype works locally but has not been tested on older laptops. Improve phrasing without claiming readiness.",
        "A clean paragraph thanking a collaborator for detailed feedback without agreeing to every suggestion. Identity example.",
        "A clumsy short application note explaining interest in a volunteer role, actual availability and two relevant experiences. Keep plain language and do not add credentials or a greeting.",
        "A long rough personal essay about learning to repair an old bicycle. Use varied concrete episodes: diagnosis, parts, a mistaken assumption, advice from a neighbor, a failed fix, safety checks, cost trade-offs, first ride and lessons stated as personal observations. Improve prose without turning it into generic advice or adding a conclusion.",
        "An awkward apology for missing a call, explaining a calendar mistake and offering to reschedule. Preserve responsibility without inventing an emergency.",
        "A repetitive but short product-feedback paragraph about a search box losing focus. Improve clarity while retaining the exact triggering steps and uncertainty about the cause.",
        "A rough internal proposal to shorten a recurring meeting, with a trial period and two concerns. Keep the proposal tentative and preserve dissent.",
        "A long uneven report about organizing a small public exhibition. Distinct sections cover submissions, labels, lighting, insurance questions, accessibility, artist permissions, installation, visitor feedback, budget and unresolved follow-up. Preserve chronology, named roles, quotes, unresolved issues and differences between observations and recommendations.",
        "An awkward personal message declining a weekend trip because the speaker needs rest. Keep warmth without inventing a health condition or promising another trip.",
        "A correct paragraph explaining a design choice and one unresolved trade-off. Identity example.",
        "A rough explanation for nontechnical colleagues about why an offline feature is useful. Use supplied examples, acknowledge synchronization limits and avoid jargon or invented guarantees.",
        "A long rough review of a local history walking tour. Include distinct stops, guide quotations, route problems, a helpful detour, accessibility observations, uncertain historical claims, booking experience, weather and a nuanced recommendation. Keep personal voice and do not add promotional wording or a summary ending.",
        "A clumsy note raising concern about a deadline that depends on another team's unfinished work. Preserve the dependency and avoid blaming the team.",
        "A rough update about a delayed parcel where tracking is inconsistent. Improve readability without claiming the parcel is lost or promising delivery.",
        "An awkward research-notes paragraph distinguishing what two interviewees said from the writer's interpretation. Preserve quotes and uncertainty exactly.",
        "A long rough retrospective of building a small home recording setup. Distinct details cover room acoustics, microphone choices, borrowed gear, monitoring, desk layout, software problems, noise, collaboration, costs and remaining questions. Keep subjective judgments and failures; no generic buying guide.",
        "An awkward message asking a neighbor to lower music after a certain hour. Preserve the request and friendly tone without turning it into a threat.",
        "A clean short paragraph describing an unfinished creative project without promising a completion date. Identity example.",
        "A rough update about revising a public document after feedback. Distinguish accepted changes, rejected suggestions and issues still under discussion.",
        "A long rough account of coordinating translations for a small open-source project. Cover terminology, contributors, review timing, interface length limits, pluralization, right-to-left testing, screenshots, disagreement resolution, release timing and unreviewed strings. Preserve who proposed what and what remains unresolved.",
        "A clumsy paragraph explaining why the writer left a club committee. Preserve their mixed feelings and do not invent conflict or motives.",
        "An already clear paragraph asking for feedback on a draft without asking the reader to rewrite it. Identity example.",
        "An awkward note about rescheduling a rehearsal because one room is unavailable. Preserve the alternatives and make clear the group has not chosen one.",
        "A long rough account of teaching a beginner programming course. Include distinct classroom incidents, setup problems, exercises, accessibility adjustments, student quotations, feedback, scheduling, assessment changes and questions for the next course. Improve readability while preserving the writer's tentative interpretation and original ending.",
    ],
}


def scenarios():
    plans = []
    for mode, briefs in BRIEFS.items():
        for index, brief in enumerate(briefs):
            split = "train" if index < 16 else "dev" if index < 20 else "validation"
            length = "long" if index % 4 == 3 else "medium" if index % 4 == 2 else "short"
            language = "en"
            if mode == "/transcribe" and index in {4, 16}:
                language = "de"
            elif mode == "/transcribe" and index in {8, 20}:
                language = "pt"
            context = {}
            if mode == "/transcribe" and index in {5, 10, 13, 22}:
                context["dictionary"] = [
                    "AuroraPad",
                    "BriarSync",
                    "settings.toml",
                    "pyproject.toml",
                    "pnpm",
                    "uv",
                ]
            if mode == "/polish" and index % 4 == 2:
                context["profile"] = (
                    "Use plain, direct prose. Preserve the writer's voice and uncertainty."
                )
            source_id = f"arka-v2-{mode[1:]}-{index + 1:02d}"
            plans.append(
                {
                    "id": source_id,
                    "source_group": source_id,
                    "mode": mode,
                    "split": split,
                    "length": length,
                    "language": language,
                    "identity": index in {1, 9, 17, 21},
                    "context": context,
                    "brief": brief,
                }
            )
    return plans
