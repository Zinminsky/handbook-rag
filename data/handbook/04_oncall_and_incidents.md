# On-Call and Incident Response

## Rotation
Engineers on the platform team rotate on-call duty weekly. Handoff happens every Monday at 10:00 CET. Each rotation has a primary and a secondary on-call engineer.

## On-call compensation
On-call engineers receive a flat €250 for each on-call week. Work done outside core hours during an incident is compensated with time off in lieu at 1.5 hours for every hour worked.

## Severity levels
SEV1 is a customer-facing outage and must be acknowledged within 5 minutes. SEV2 is a major degradation and must be acknowledged within 15 minutes. SEV3 is a minor issue and is handled on the next business day.

## Escalation
If the primary on-call engineer has not acknowledged an alert within 15 minutes, it escalates automatically to the secondary. During a SEV1 the incident commander posts a status page update every 30 minutes.

## Postmortems
A blameless postmortem is required for every SEV1 and SEV2 incident. It must be published within 5 business days of resolution.
