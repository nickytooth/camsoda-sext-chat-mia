"use client";

import ProfileGallery from "./ProfileGallery";
import {
  Heart,
  User,
  Dumbbell,
  Globe,
  Languages,
  HeartHandshake,
  Briefcase,
  Sparkles,
  Wine,
} from "lucide-react";

interface ProfileData {
  age: string;
  body: string;
  ethnicity: string;
  language: string;
  relationship: string;
  occupation: string;
  hobbies: string;
  personality: string;
}

interface Props {
  name: string;
  tagline: string;
  bio?: string;
  profile: ProfileData;
}

const PROFILE_FIELDS: {
  key: keyof ProfileData;
  label: string;
  icon: React.ReactNode;
}[] = [
  { key: "age", label: "AGE", icon: <User size={18} /> },
  { key: "body", label: "BODY", icon: <Dumbbell size={18} /> },
  { key: "ethnicity", label: "ETHNICITY", icon: <Globe size={18} /> },
  { key: "language", label: "LANGUAGE", icon: <Languages size={18} /> },
  { key: "relationship", label: "RELATIONSHIP", icon: <HeartHandshake size={18} /> },
  { key: "occupation", label: "OCCUPATION", icon: <Briefcase size={18} /> },
  { key: "hobbies", label: "HOBBIES", icon: <Wine size={18} /> },
  { key: "personality", label: "PERSONALITY", icon: <Sparkles size={18} /> },
];

export default function ProfileSidebar({ name, tagline, bio, profile }: Props) {
  return (
    <aside className="w-[320px] flex-shrink-0 bg-[var(--sidebar-bg)] border-l border-[var(--border)] overflow-y-auto h-full">
      {/* Profile media — video + photos carousel (arrows + dots) */}
      <ProfileGallery />

      {/* Name + tagline */}
      <div className="px-5 pt-4 pb-3">
        <div className="flex items-center gap-2">
          <h2 className="text-xl font-bold text-white">{name}</h2>
          <Heart size={18} className="text-pink-400" />
        </div>
        <p className="text-[13px] text-[var(--muted)] mt-1.5 leading-relaxed">
          {tagline}
        </p>
      </div>

      {/* Divider */}
      <div className="mx-5 border-t border-[var(--border)]" />

      {/* Bio */}
      {bio && (
        <>
          <div className="px-5 pt-4 pb-1">
            <h3 className="text-[13px] font-semibold text-[var(--muted)] mb-2">
              Bio
            </h3>
            <p className="text-[13px] text-white/90 leading-relaxed">{bio}</p>
          </div>
          <div className="mx-5 mt-4 border-t border-[var(--border)]" />
        </>
      )}

      {/* About me */}
      <div className="px-5 pt-4 pb-6">
        <h3 className="text-[13px] font-semibold text-[var(--muted)] mb-4">
          About me:
        </h3>
        <div className="grid grid-cols-2 gap-4">
          {PROFILE_FIELDS.map((field) => (
            <div key={field.key} className="flex items-start gap-2.5">
              <div className="text-[var(--muted)] mt-0.5">{field.icon}</div>
              <div>
                <div className="text-[10px] text-[var(--muted)] uppercase tracking-wider">
                  {field.label}
                </div>
                <div className="text-[13px] text-white font-medium mt-0.5">
                  {profile[field.key] || "—"}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}
