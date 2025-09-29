# Digital Pain Translator - Design Guidelines

## Design Approach: Design System (Medical/Healthcare Focus)
**Selected System**: Material Design with healthcare adaptations
**Justification**: This utility-focused healthcare application requires clinical precision, accessibility, and trust-building design elements. Material Design provides the structured approach needed for medical interfaces while ensuring accessibility compliance.

## Core Design Elements

### A. Color Palette
**Primary Colors**:
- Medical Blue: `210 90% 25%` (dark mode: `210 70% 85%`)
- Clinical White: `0 0% 98%` (dark mode: `210 15% 12%`)

**Semantic Colors**:
- Success/Low Pain: `120 60% 35%` (dark mode: `120 50% 70%`)
- Warning/Medium Pain: `35 85% 50%` (dark mode: `35 70% 70%`)
- Error/High Pain: `0 75% 45%` (dark mode: `0 60% 75%`)
- Neutral/Info: `210 25% 55%` (dark mode: `210 30% 75%`)

### B. Typography
**Primary Font**: Inter (Google Fonts)
**Hierarchy**:
- Headings: 600 weight, larger sizes for pain scores
- Body text: 400 weight, high contrast for readability
- Data/Numbers: 500 weight, monospace for consistency
- Clinical context requires 16px minimum for body text

### C. Layout System
**Tailwind Spacing**: Primary units of 2, 4, 6, 8, 12, 16
- Component spacing: `p-4`, `m-6`
- Section spacing: `py-8`, `px-6`
- Grid gaps: `gap-4`, `gap-6`
- Consistent 8px baseline grid

### D. Component Library

**Core Interface Components**:
- **Camera Viewport**: Rounded container with landmark overlay, consent indicators
- **Caregiver Controls**: Clean slider components with clear labeling and real-time feedback
- **Pain Score Display**: Large, prominent numeric display with confidence percentage
- **Feature Explanation Cards**: Structured cards showing contributing factors with visual weights
- **Data Export Panel**: Simple table view with CSV download functionality

**Navigation**: 
- Minimal header with app title and settings
- Tab-based navigation for different assessment phases
- Clear progress indicators during analysis

**Forms**:
- High-contrast form inputs with clear labels
- Slider components with numeric values and color coding
- Large, accessible button targets (minimum 44px touch targets)

**Data Displays**:
- Real-time charts for facial feature tracking
- Color-coded pain level indicators (green/yellow/red progression)
- Confidence meters with percentage displays
- Historical data tables with timestamp sorting

**Overlays**:
- Privacy consent modal with clear, accessible language
- Loading states during facial analysis
- Error states with clear recovery instructions

### E. Healthcare-Specific Design Considerations

**Trust & Privacy**:
- Prominent privacy indicators showing local-only processing
- Clear consent language and camera permissions
- Professional, clinical aesthetic without being sterile
- Subtle shadows and borders for definition without distraction

**Accessibility**:
- High contrast ratios (minimum 4.5:1) for all text
- Focus indicators for keyboard navigation
- Screen reader optimization for all interactive elements
- Color-blind friendly palette with pattern/texture alternatives

**Clinical Workflow**:
- Clear visual hierarchy prioritizing pain score results
- Explainable AI section with digestible factor breakdowns
- Quick action buttons for common clinical responses
- Efficient data entry with smart defaults

**Visual Hierarchy**:
- Pain score as dominant visual element (large typography, central placement)
- Secondary information (confidence, factors) clearly subordinated
- Caregiver inputs grouped logically with clear section divisions
- Camera feed prominent but not overwhelming

## Images
No large hero images required. The application is utility-focused with the primary visual being the real-time camera feed showing facial landmark detection overlays. Any illustrative content should be simple icons or diagrams explaining the pain assessment process.